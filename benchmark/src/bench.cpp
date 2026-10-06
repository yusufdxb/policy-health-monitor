// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_bench/bench.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/numpy_random.hpp"
#include "phm_tools/torch_mlp.hpp"

namespace phm_bench
{

namespace
{

using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecD = Eigen::VectorXd;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

MatD to_eigen(const Matrix & m)
{
  MatD out(static_cast<Eigen::Index>(m.rows), static_cast<Eigen::Index>(m.cols));
  std::copy(m.data.begin(), m.data.end(), out.data());
  return out;
}

// x.mean(axis=0) on a C-contiguous (N, D) array: per column, rows summed in
// order, then divided by N.
VecD column_mean(const MatD & x)
{
  VecD acc = VecD::Zero(x.cols());
  for (Eigen::Index r = 0; r < x.rows(); ++r) {
    for (Eigen::Index c = 0; c < x.cols(); ++c) {
      acc[c] += x(r, c);
    }
  }
  for (Eigen::Index c = 0; c < x.cols(); ++c) {
    acc[c] /= static_cast<double>(x.rows());
  }
  return acc;
}

// Pairwise (NumPy-order) sum of each row's values.
template<typename F>
std::vector<double> row_sums(Eigen::Index rows, Eigen::Index cols, F value)
{
  std::vector<double> out(static_cast<std::size_t>(rows));
  std::vector<double> tmp(static_cast<std::size_t>(cols));
  for (Eigen::Index r = 0; r < rows; ++r) {
    for (Eigen::Index c = 0; c < cols; ++c) {
      tmp[static_cast<std::size_t>(c)] = value(r, c);
    }
    out[static_cast<std::size_t>(r)] = phm_core::numpy_sum(tmp.data(), tmp.size());
  }
  return out;
}

// x_t = mean + rho (x_{t-1} - mean) + innov * (L @ eps_t), first x = mean + L @ eps.
MatD ar1_gaussian(
  int n, const VecD & mean, const MatD & cov, double rho, phm_core::numpy_random::Generator & rng)
{
  const Eigen::Index dim = mean.size();
  const MatD chol = MatD(cov.llt().matrixL());
  const double innov = std::sqrt(std::max(1.0 - rho * rho, 1e-9));
  MatD out(n, dim);
  VecD z(dim);
  for (Eigen::Index i = 0; i < dim; ++i) {
    z[i] = rng.normal(0.0, 1.0);
  }
  VecD x = mean + chol * z;
  for (int t = 0; t < n; ++t) {
    for (Eigen::Index i = 0; i < dim; ++i) {
      z[i] = rng.normal(0.0, 1.0);
    }
    const VecD eps = chol * z;
    for (Eigen::Index i = 0; i < dim; ++i) {
      x[i] = (mean[i] + rho * (x[i] - mean[i])) + innov * eps[i];
    }
    out.row(t) = x.transpose();
  }
  return out;
}

Matrix to_matrix(const MatD & m)
{
  Matrix out(static_cast<std::size_t>(m.rows()), static_cast<std::size_t>(m.cols()));
  std::copy(m.data(), m.data() + m.size(), out.data.begin());
  return out;
}

struct Gaussian
{
  VecD mu;
  MatD prec;
};

// Empirical mean and ridge-regularized precision (trace-relative shrinkage).
Gaussian fit_gaussian(const MatD & f, double reg)
{
  Gaussian g;
  g.mu = column_mean(f);
  const MatD x = f.rowwise() - g.mu.transpose();
  MatD cov = (x.transpose() * x) / static_cast<double>(std::max<Eigen::Index>(f.rows() - 1, 1));
  std::vector<double> diag(static_cast<std::size_t>(cov.rows()));
  for (Eigen::Index i = 0; i < cov.rows(); ++i) {
    diag[static_cast<std::size_t>(i)] = cov(i, i);
  }
  const double trace_mean = cov.rows() ?
    phm_core::numpy_sum(diag.data(), diag.size()) / static_cast<double>(cov.rows()) : 1.0;
  const double ridge = reg * trace_mean + 1e-6;
  for (Eigen::Index i = 0; i < cov.rows(); ++i) {
    cov(i, i) += ridge;
  }
  g.prec = cov.inverse();
  return g;
}

std::vector<double> mahalanobis_with(const Gaussian & g, const MatD & test)
{
  const MatD x = test.rowwise() - g.mu.transpose();
  const MatD xp = x * g.prec;
  return row_sums(x.rows(), x.cols(), [&](Eigen::Index r, Eigen::Index c) {
      return xp(r, c) * x(r, c);
    });
}

void clean(
  const std::vector<double> & s, const std::vector<int> & y, std::vector<double> & so,
  std::vector<int> & yo)
{
  so.clear();
  yo.clear();
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (std::isfinite(s[i])) {
      so.push_back(s[i]);
      yo.push_back(y[i]);
    }
  }
}

bool two_classes(const std::vector<int> & y)
{
  bool pos = false;
  bool neg = false;
  for (int v : y) {
    pos |= v == 1;
    neg |= v == 0;
  }
  return pos && neg;
}

// Descending stable order (np.argsort(-s, kind="mergesort")) and the
// threshold-point cumulative counts.
struct Curve
{
  std::vector<double> tps;
  std::vector<double> fps;
};

Curve descending_curve(const std::vector<double> & s, const std::vector<int> & y)
{
  std::vector<std::size_t> order(s.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&s](std::size_t a, std::size_t b) {
      return -s[a] < -s[b];
    });
  Curve c;
  double tp = 0.0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    tp += y[order[i]] == 1 ? 1.0 : 0.0;
    const bool last = i + 1 == order.size();
    if (last || s[order[i + 1]] != s[order[i]]) {
      c.tps.push_back(tp);
      c.fps.push_back(1.0 + static_cast<double>(i) - tp);
    }
  }
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------
Streams generate_stream(const StreamSpec & spec)
{
  phm_core::numpy_random::Generator rng(spec.seed);
  const int dim = spec.dim;
  const VecD mean = VecD::Zero(dim);
  // Random SPD covariance: a = N(0,1)^(dim x dim) / sqrt(dim); a a^T + 0.5 I.
  MatD a(dim, dim);
  const double sq = std::sqrt(static_cast<double>(dim));
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    a.data()[i] = rng.normal(0.0, 1.0) / sq;
  }
  MatD cov = a * a.transpose();
  for (int i = 0; i < dim; ++i) {
    cov(i, i) += 0.5;
  }
  cov *= spec.in_dist_scale * spec.in_dist_scale;

  Streams out;
  const MatD id = ar1_gaussian(spec.n_id, mean, cov, spec.ar_rho, rng);
  out.id = to_matrix(id);
  if (spec.ood_mode == "collapse") {
    // Frozen near a single point: the last ID state plus tiny noise.
    MatD ood(spec.n_ood, dim);
    for (int t = 0; t < spec.n_ood; ++t) {
      for (int i = 0; i < dim; ++i) {
        ood(t, i) = id(spec.n_id - 1, i) + rng.normal(0.0, spec.ood_scale);
      }
    }
    out.ood = to_matrix(ood);
  } else if (spec.ood_mode == "shift") {
    VecD shifted(dim);
    for (int i = 0; i < dim; ++i) {
      shifted[i] = mean[i] + spec.ood_shift * std::sqrt(cov(i, i));
    }
    out.ood = to_matrix(ar1_gaussian(spec.n_ood, shifted, 0.5 * cov, spec.ar_rho, rng));
  } else {
    throw std::invalid_argument("unknown ood_mode '" + spec.ood_mode + "'");
  }
  return out;
}

double rolling_spread_trace(const Matrix & frames, std::size_t window)
{
  std::vector<double> vals;
  for (double s : phm_core::rolling_spread(frames, window)) {
    if (!std::isnan(s)) {
      vals.push_back(s);
    }
  }
  return vals.empty() ? kNaN : phm_core::numpy_mean(vals.data(), vals.size());
}

// ---------------------------------------------------------------------------
// Detectors
// ---------------------------------------------------------------------------
std::vector<double> phm_scores(
  const Matrix & fid, const Matrix & test, std::size_t window, double percentile)
{
  const double thr = phm_core::calibrate_threshold(phm_core::rolling_spread(fid, window),
      percentile);
  std::vector<double> out = phm_core::rolling_spread(test, window);
  for (double & s : out) {
    s = thr - s;
  }
  return out;
}

std::vector<double> mahalanobis(const Matrix & fid, const Matrix & test, double reg)
{
  return mahalanobis_with(fit_gaussian(to_eigen(fid), reg), to_eigen(test));
}

std::vector<double> relative_mahalanobis(const Matrix & fid, const Matrix & test, double reg)
{
  const MatD f = to_eigen(fid);
  const MatD t = to_eigen(test);
  const std::vector<double> m_id = mahalanobis_with(fit_gaussian(f, reg), t);

  // Two-component k-means background fit, centers from RandomState(0).
  constexpr int kComponents = 2;
  phm_core::numpy_random::RandomState rs(0);
  const auto idx = rs.choice_without_replacement(static_cast<std::size_t>(f.rows()), kComponents);
  MatD centers(kComponents, f.cols());
  for (int k = 0; k < kComponents; ++k) {
    centers.row(k) = f.row(idx[static_cast<std::size_t>(k)]);
  }
  std::vector<int> assign(static_cast<std::size_t>(f.rows()), 0);
  std::vector<double> sq(static_cast<std::size_t>(f.cols()));
  for (int iter = 0; iter < 20; ++iter) {
    for (Eigen::Index r = 0; r < f.rows(); ++r) {
      double best = 0.0;
      int arg = 0;
      for (int k = 0; k < kComponents; ++k) {
        for (Eigen::Index c = 0; c < f.cols(); ++c) {
          const double d = f(r, c) - centers(k, c);
          sq[static_cast<std::size_t>(c)] = d * d;
        }
        const double dist = phm_core::numpy_sum(sq.data(), sq.size());
        if (k == 0 || dist < best) {  // argmin: first minimum wins
          best = dist;
          arg = k;
        }
      }
      assign[static_cast<std::size_t>(r)] = arg;
    }
    MatD next = centers;
    for (int k = 0; k < kComponents; ++k) {
      VecD acc = VecD::Zero(f.cols());
      int count = 0;
      for (Eigen::Index r = 0; r < f.rows(); ++r) {
        if (assign[static_cast<std::size_t>(r)] == k) {
          for (Eigen::Index c = 0; c < f.cols(); ++c) {
            acc[c] += f(r, c);
          }
          ++count;
        }
      }
      if (count > 0) {
        next.row(k) = (acc / static_cast<double>(count)).transpose();
      }
    }
    // np.allclose(new, centers, atol=1e-6): |a - b| <= 1e-6 + 1e-5 * |b|.
    bool close = true;
    for (Eigen::Index i = 0; i < next.size() && close; ++i) {
      close = std::fabs(next.data()[i] - centers.data()[i]) <=
        1e-6 + 1e-5 * std::fabs(centers.data()[i]);
    }
    centers = next;
    if (close) {
      break;
    }
  }
  std::vector<double> comp_min(static_cast<std::size_t>(t.rows()),
    std::numeric_limits<double>::infinity());
  for (int k = 0; k < kComponents; ++k) {
    std::vector<Eigen::Index> rows;
    for (Eigen::Index r = 0; r < f.rows(); ++r) {
      if (assign[static_cast<std::size_t>(r)] == k) {
        rows.push_back(r);
      }
    }
    MatD cluster = f;
    if (rows.size() >= 2) {
      cluster.resize(static_cast<Eigen::Index>(rows.size()), f.cols());
      for (std::size_t i = 0; i < rows.size(); ++i) {
        cluster.row(static_cast<Eigen::Index>(i)) = f.row(rows[i]);
      }
    }
    const auto m = mahalanobis_with(fit_gaussian(cluster, reg), t);
    for (std::size_t i = 0; i < m.size(); ++i) {
      comp_min[i] = std::min(comp_min[i], m[i]);
    }
  }
  std::vector<double> out(m_id.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = comp_min[i] - m_id[i];
  }
  return out;
}

std::vector<double> knn_distance(const Matrix & fid, const Matrix & test, int k, bool normalize)
{
  MatD a = to_eigen(fid);
  MatD b = to_eigen(test);
  if (normalize) {
    for (MatD * m : {&a, &b}) {
      const auto n = row_sums(m->rows(), m->cols(), [m](Eigen::Index r, Eigen::Index c) {
          return (*m)(r, c) * (*m)(r, c);
        });
      for (Eigen::Index r = 0; r < m->rows(); ++r) {
        double norm = std::sqrt(n[static_cast<std::size_t>(r)]);
        if (norm == 0.0) {
          norm = 1.0;
        }
        m->row(r) /= norm;
      }
    }
  }
  k = std::min<int>(k, static_cast<int>(a.rows()));
  const auto a_norm = row_sums(a.rows(), a.cols(), [&a](Eigen::Index r, Eigen::Index c) {
      return a(r, c) * a(r, c);
    });
  const auto b_norm = row_sums(b.rows(), b.cols(), [&b](Eigen::Index r, Eigen::Index c) {
      return b(r, c) * b(r, c);
    });
  const MatD cross = (2.0 * b) * a.transpose();
  std::vector<double> out(static_cast<std::size_t>(b.rows()));
  std::vector<double> d2(static_cast<std::size_t>(a.rows()));
  for (Eigen::Index i = 0; i < b.rows(); ++i) {
    for (Eigen::Index j = 0; j < a.rows(); ++j) {
      const double v = b_norm[static_cast<std::size_t>(i)] + a_norm[static_cast<std::size_t>(j)] -
        cross(i, j);
      d2[static_cast<std::size_t>(j)] = std::max(v, 0.0);
    }
    std::nth_element(d2.begin(), d2.begin() + (k - 1), d2.end());
    out[static_cast<std::size_t>(i)] = std::sqrt(d2[static_cast<std::size_t>(k - 1)]);
  }
  return out;
}

std::vector<double> rnd_closed_form(
  const Matrix & fid, const Matrix & test, int proj_dim, double reg, uint32_t seed)
{
  const MatD f = to_eigen(fid);
  const MatD t = to_eigen(test);
  const Eigen::Index d = f.cols();
  phm_core::numpy_random::RandomState rs(seed);
  MatD w_t(d, proj_dim);
  const double sd_d = std::sqrt(static_cast<double>(d));
  for (Eigen::Index i = 0; i < w_t.size(); ++i) {
    w_t.data()[i] = rs.gauss() / sd_d;
  }
  MatD v_t(proj_dim, proj_dim);
  const double sd_p = std::sqrt(static_cast<double>(proj_dim));
  for (Eigen::Index i = 0; i < v_t.size(); ++i) {
    v_t.data()[i] = rs.gauss() / sd_p;
  }
  const auto target = [&](const MatD & x) {
      return MatD((x * w_t).cwiseMax(0.0) * v_t);
    };
  const MatD t_id = target(f);

  // Standardize on ID statistics (np.mean / np.std, ddof 0).
  const VecD mu = column_mean(f);
  VecD sd = VecD::Zero(d);
  for (Eigen::Index r = 0; r < f.rows(); ++r) {
    for (Eigen::Index c = 0; c < d; ++c) {
      const double x = f(r, c) - mu[c];
      sd[c] += x * x;
    }
  }
  for (Eigen::Index c = 0; c < d; ++c) {
    sd[c] = std::sqrt(sd[c] / static_cast<double>(f.rows()));
    if (sd[c] == 0.0) {
      sd[c] = 1.0;
    }
  }
  const auto feat = [&](const MatD & x) {
      MatD phi(x.rows(), d + 1);
      for (Eigen::Index r = 0; r < x.rows(); ++r) {
        for (Eigen::Index c = 0; c < d; ++c) {
          phi(r, c) = (x(r, c) - mu[c]) / sd[c];
        }
        phi(r, d) = 1.0;
      }
      return phi;
    };
  const MatD phi_id = feat(f);
  MatD gram = phi_id.transpose() * phi_id;
  for (Eigen::Index i = 0; i < gram.rows(); ++i) {
    gram(i, i) += reg;
  }
  const MatD w_pred = gram.partialPivLu().solve(MatD(phi_id.transpose() * t_id));
  const MatD resid = target(t) - feat(t) * w_pred;
  auto err = row_sums(resid.rows(), resid.cols(), [&resid](Eigen::Index r, Eigen::Index c) {
      return resid(r, c) * resid(r, c);
    });
  for (double & e : err) {
    e /= static_cast<double>(resid.cols());
  }
  return err;
}

RndMlpResult rnd_mlp(const Matrix & fid, const Matrix & test, int epochs, uint64_t seed)
{
  using phm_tools::torch_mlp::MatrixF;
  const Eigen::Index d = static_cast<Eigen::Index>(fid.cols);
  const MatrixF x_id = to_eigen(fid).cast<float>();
  const MatrixF x_test = to_eigen(test).cast<float>();
  // Standardize on ID stats: mean and unbiased std (torch.std), clamp_min(1e-6).
  Eigen::RowVectorXd mu = x_id.cast<double>().colwise().mean();
  Eigen::RowVectorXd var = Eigen::RowVectorXd::Zero(d);
  for (Eigen::Index r = 0; r < x_id.rows(); ++r) {
    var += (x_id.row(r).cast<double>() - mu).array().square().matrix();
  }
  var /= static_cast<double>(std::max<Eigen::Index>(x_id.rows() - 1, 1));
  const Eigen::RowVectorXf muf = mu.cast<float>();
  const Eigen::RowVectorXf sdf = var.array().sqrt().max(1e-6).matrix().cast<float>();
  const auto standardize = [&](const MatrixF & x) {
      MatrixF out = x;
      for (Eigen::Index r = 0; r < out.rows(); ++r) {
        out.row(r) = ((x.row(r) - muf).array() / sdf.array()).matrix();
      }
      return out;
    };
  const MatrixF id_n = standardize(x_id);
  const MatrixF test_n = standardize(x_test);

  constexpr int kProj = 128;
  auto gen = phm_tools::torch_mlp::manual_seed(seed);
  using phm_tools::torch_mlp::Activation;
  const phm_tools::torch_mlp::Mlp target({static_cast<int>(d), kProj, kProj}, Activation::kReLU,
    gen);
  phm_tools::torch_mlp::Mlp predictor({static_cast<int>(d), kProj, kProj}, Activation::kReLU, gen);
  const MatrixF t_id = target.forward(id_n);
  RndMlpResult res;
  for (int e = 0; e < epochs; ++e) {
    res.final_train_mse = predictor.train_step(id_n, t_id, 1e-3);
  }
  const MatrixF diff = target.forward(test_n) - predictor.forward(test_n);
  res.scores.resize(static_cast<std::size_t>(diff.rows()));
  for (Eigen::Index r = 0; r < diff.rows(); ++r) {
    res.scores[static_cast<std::size_t>(r)] =
      static_cast<double>(diff.row(r).array().square().mean());
  }
  return res;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
double auroc(const std::vector<double> & scores, const std::vector<int> & labels)
{
  std::vector<double> s;
  std::vector<int> y;
  clean(scores, labels, s, y);
  if (!two_classes(y)) {
    return kNaN;
  }
  std::vector<std::size_t> order(s.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&s](std::size_t a, std::size_t b) {
      return s[a] < s[b];
    });
  std::vector<double> ranks(s.size());
  std::size_t i = 0;
  while (i < order.size()) {
    std::size_t j = i;
    while (j + 1 < order.size() && s[order[j + 1]] == s[order[i]]) {
      ++j;
    }
    const double avg = static_cast<double>(i + j) / 2.0 + 1.0;
    for (std::size_t k = i; k <= j; ++k) {
      ranks[order[k]] = avg;
    }
    i = j + 1;
  }
  double n_pos = 0.0;
  double n_neg = 0.0;
  std::vector<double> pos_ranks;
  for (std::size_t k = 0; k < y.size(); ++k) {
    if (y[k] == 1) {
      n_pos += 1.0;
      pos_ranks.push_back(ranks[k]);
    } else {
      n_neg += 1.0;
    }
  }
  const double sum_pos = phm_core::numpy_sum(pos_ranks.data(), pos_ranks.size());
  return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
}

double aupr(const std::vector<double> & scores, const std::vector<int> & labels)
{
  std::vector<double> s;
  std::vector<int> y;
  clean(scores, labels, s, y);
  if (!two_classes(y)) {
    return kNaN;
  }
  double p = 0.0;
  for (int v : y) {
    p += v == 1 ? 1.0 : 0.0;
  }
  const Curve c = descending_curve(s, y);
  std::vector<double> terms(c.tps.size());
  double recall_prev = 0.0;
  for (std::size_t k = 0; k < c.tps.size(); ++k) {
    const double precision = c.tps[k] / std::max(c.tps[k] + c.fps[k], 1e-12);
    const double recall = c.tps[k] / p;
    terms[k] = precision * (recall - recall_prev);
    recall_prev = recall;
  }
  return phm_core::numpy_sum(terms.data(), terms.size());
}

double fpr_at_tpr(
  const std::vector<double> & scores, const std::vector<int> & labels, double tpr_target)
{
  std::vector<double> s;
  std::vector<int> y;
  clean(scores, labels, s, y);
  if (!two_classes(y)) {
    return kNaN;
  }
  double p = 0.0;
  double n = 0.0;
  for (int v : y) {
    (v == 1 ? p : n) += 1.0;
  }
  const Curve c = descending_curve(s, y);
  // tpr = [0, tps / p]; first index with tpr >= target (searchsorted left).
  if (0.0 >= tpr_target) {
    return 0.0;
  }
  for (std::size_t k = 0; k < c.tps.size(); ++k) {
    if (c.tps[k] / p >= tpr_target) {
      return c.fps[k] / n;
    }
  }
  return kNaN;
}

Ci bootstrap_ci(
  const MetricFn & metric, const std::vector<double> & scores, const std::vector<int> & labels,
  int n_bootstrap, double ci, uint32_t seed)
{
  std::vector<double> s;
  std::vector<int> y;
  clean(scores, labels, s, y);
  std::vector<std::size_t> pos;
  std::vector<std::size_t> neg;
  for (std::size_t i = 0; i < y.size(); ++i) {
    (y[i] == 1 ? pos : neg).push_back(i);
  }
  if (pos.empty() || neg.empty()) {
    return {kNaN, kNaN, kNaN};
  }
  phm_core::numpy_random::RandomState rs(seed);
  std::vector<int64_t> rp(pos.size());
  std::vector<int64_t> rn(neg.size());
  std::vector<double> bs(pos.size() + neg.size());
  std::vector<int> by(pos.size() + neg.size());
  std::vector<double> vals;
  vals.reserve(static_cast<std::size_t>(n_bootstrap));
  for (int b = 0; b < n_bootstrap; ++b) {
    rs.randint(0, static_cast<int64_t>(pos.size()), rp.data(), rp.size());
    rs.randint(0, static_cast<int64_t>(neg.size()), rn.data(), rn.size());
    std::size_t k = 0;
    for (int64_t r : rp) {
      bs[k] = s[pos[static_cast<std::size_t>(r)]];
      by[k++] = 1;
    }
    for (int64_t r : rn) {
      bs[k] = s[neg[static_cast<std::size_t>(r)]];
      by[k++] = 0;
    }
    const double v = metric(bs, by);
    if (std::isfinite(v)) {
      vals.push_back(v);
    }
  }
  if (vals.empty()) {
    return {kNaN, kNaN, kNaN};
  }
  const double alpha = (1.0 - ci) / 2.0;
  return {
    phm_core::numpy_mean(vals.data(), vals.size()), phm_core::quantile(vals, alpha),
    phm_core::quantile(vals, 1.0 - alpha)};
}

}  // namespace phm_bench
