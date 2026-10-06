// Copyright 2026 Yusuf Guenena. MIT License.
// Severity bands, normalize(), the Hysteresis counter and the Detector
// interface (ported from phm_core/tests/test_{severity,hysteresis,detector}.py).
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "phm_core/detector.hpp"
#include "phm_core/hysteresis.hpp"
#include "phm_core/severity.hpp"

using phm_core::classify;
using phm_core::Hysteresis;
using phm_core::normalize;

TEST(Severity, ClassifyBands)
{
  using phm_core::ACTION_HOLD;
  using phm_core::ACTION_LOG_ONLY;
  using phm_core::ACTION_NONE;
  using phm_core::ACTION_STOP_AND_HOLD;
  const std::vector<std::tuple<double, uint8_t, uint8_t>> cases = {
    {0.0, phm_core::STATE_OK, ACTION_NONE},
    {0.10, phm_core::STATE_OK, ACTION_NONE},
    {0.249, phm_core::STATE_OK, ACTION_NONE},
    {0.25, phm_core::STATE_DEGRADED, ACTION_LOG_ONLY},
    {0.40, phm_core::STATE_DEGRADED, ACTION_LOG_ONLY},
    {0.499, phm_core::STATE_DEGRADED, ACTION_LOG_ONLY},
    {0.50, phm_core::STATE_INTERVENE, ACTION_HOLD},
    {0.70, phm_core::STATE_INTERVENE, ACTION_HOLD},
    {0.799, phm_core::STATE_INTERVENE, ACTION_HOLD},
    {0.80, phm_core::STATE_STOP, ACTION_STOP_AND_HOLD},
    {0.95, phm_core::STATE_STOP, ACTION_STOP_AND_HOLD},
    {1.0, phm_core::STATE_STOP, ACTION_STOP_AND_HOLD},
  };
  for (const auto & [score, state, action] : cases) {
    const auto sev = classify(score);
    EXPECT_EQ(sev.state, state) << score;
    EXPECT_EQ(sev.suggested_action, action) << score;
  }
}

TEST(Severity, BandEdgesAreInclusiveAtLowerEdge)
{
  EXPECT_EQ(classify(phm_core::DEGRADED_THRESHOLD).state, phm_core::STATE_DEGRADED);
  EXPECT_EQ(classify(phm_core::INTERVENE_THRESHOLD).state, phm_core::STATE_INTERVENE);
  EXPECT_EQ(classify(phm_core::STOP_THRESHOLD).state, phm_core::STATE_STOP);
}

TEST(Severity, ClassifyClampsOutOfRange)
{
  const auto low = classify(-0.5);
  EXPECT_EQ(low.score, 0.0);
  EXPECT_EQ(low.state, phm_core::STATE_OK);
  const auto high = classify(1.7);
  EXPECT_EQ(high.score, 1.0);
  EXPECT_EQ(high.state, phm_core::STATE_STOP);
}

TEST(Severity, NormalizeBothDirections)
{
  EXPECT_DOUBLE_EQ(normalize(5.0, 0.0, 10.0), 0.5);
  EXPECT_DOUBLE_EQ(normalize(2.5, 0.0, 10.0), 0.25);
  EXPECT_EQ(normalize(-3.0, 0.0, 10.0), 0.0);
  EXPECT_EQ(normalize(99.0, 0.0, 10.0), 1.0);
  // Rolling-spread collapse: a LOW value is unhealthy.
  EXPECT_DOUBLE_EQ(normalize(0.5, 1.0, 0.0), 0.5);
  EXPECT_DOUBLE_EQ(normalize(0.0, 1.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(normalize(1.0, 1.0, 0.0), 0.0);
  EXPECT_EQ(normalize(2.0, 1.0, 0.0), 0.0);
  EXPECT_THROW(normalize(1.0, 3.0, 3.0), std::invalid_argument);
  // Fully collapsed spread is worst case through normalize + classify.
  EXPECT_EQ(classify(normalize(0.0, 1.0, 0.0)).state, phm_core::STATE_STOP);
}

TEST(HysteresisTest, FiresOnlyAfterMinConsecutive)
{
  Hysteresis h(3);
  EXPECT_FALSE(h.observe(true));
  EXPECT_FALSE(h.observe(true));
  EXPECT_TRUE(h.observe(true));
  EXPECT_TRUE(h.observe(true));  // stays fired while the run continues
}

TEST(HysteresisTest, HealthySampleResetsTheRun)
{
  Hysteresis h(3);
  h.observe(true);
  h.observe(true);
  EXPECT_EQ(h.count(), 2);
  EXPECT_FALSE(h.observe(false));
  EXPECT_EQ(h.count(), 0);
  EXPECT_FALSE(h.observe(true));
  EXPECT_FALSE(h.observe(true));
  EXPECT_TRUE(h.observe(true));
}

TEST(HysteresisTest, IntermittentViolationsNeverFire)
{
  Hysteresis h(3);
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(h.observe(true));
    EXPECT_FALSE(h.observe(false));
  }
}

TEST(HysteresisTest, MinOneFiresImmediatelyAndResetWorks)
{
  Hysteresis h(1);
  EXPECT_TRUE(h.observe(true));
  EXPECT_FALSE(h.observe(false));
  EXPECT_TRUE(h.observe(true));
  Hysteresis h2(2);
  h2.observe(true);
  h2.reset();
  EXPECT_EQ(h2.count(), 0);
  EXPECT_FALSE(h2.observe(true));
  EXPECT_EQ(Hysteresis(4).min_consecutive(), 4);
}

TEST(HysteresisTest, RejectsMinConsecutiveBelowOne)
{
  EXPECT_THROW(Hysteresis(0), std::invalid_argument);
  EXPECT_THROW(Hysteresis(-1), std::invalid_argument);
}

namespace
{
class SpreadDetector : public phm_core::Detector<double>
{
public:
  SpreadDetector()
  : Detector("phm_ood", "/policy/embedding") {}
  std::optional<phm_core::VerdictData> update(const double & sample) override
  {
    if (sample < 0.5) {
      phm_core::VerdictData v;
      v.source = name();
      v.score = 0.9;
      v.violating = true;
      v.reason = "spread below floor";
      v.suggested_action = phm_core::ACTION_HOLD;
      return v;
    }
    return std::nullopt;
  }
};
}  // namespace

TEST(DetectorInterface, ConcreteDetectorReturnsVerdictOrNothing)
{
  SpreadDetector det;
  EXPECT_EQ(det.target_topic(), "/policy/embedding");
  EXPECT_FALSE(det.update(0.8).has_value());
  const auto v = det.update(0.1);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->source, "phm_ood");
  EXPECT_TRUE(v->violating);
  EXPECT_EQ(v->suggested_action, phm_core::ACTION_HOLD);
  EXPECT_EQ(phm_core::VerdictData{}.suggested_action, phm_core::ACTION_NONE);
}
