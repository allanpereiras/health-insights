import operator
from datetime import datetime, time, timedelta
from itertools import chain

import pytz
from dateutil.relativedelta import relativedelta
from django.db import IntegrityError, models
from django.test import TestCase
from faker import Faker
from insights import models as insight_models
from insights.tests import factories

faker = Faker()


class TestDataDevice(TestCase):
    """Test module for Data Device model"""

    def setUp(self) -> None:
        self.instance = factories.DataDeviceFactory()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)

    def test_name_max_length(self):
        max_length = self.instance._meta.get_field("name").max_length
        self.assertEquals(max_length, 50)

    def test_name_not_null_empty(self):
        self.instance.name = None
        self.assertRaises(IntegrityError, self.instance.save)
        # self.instance.name = ""
        # self.assertRaises(IntegrityError, self.instance.save)

    def test_version_max_length(self):
        max_length = self.instance._meta.get_field("version").max_length
        self.assertEquals(max_length, 16)

    def test_version_not_null_empty(self):
        self.instance.version = None
        self.assertRaises(IntegrityError, self.instance.save)
        # self.instance.version = ""
        # self.assertRaises(IntegrityError, self.instance.save)


class TestStatistic(TestCase):
    """Test cases for model Statistic"""

    fixtures = ["insights/tests/fixtures/Statistic.json"]

    def setUp(self) -> None:
        self.instance = factories.StatisticFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)


class TestPatientDevice(TestCase):
    """Test cases for model PatientDevice"""

    def setUp(self) -> None:
        self.instance = factories.PatientDeviceFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.id))


class TestDataDistributor(TestCase):
    """Test cases for model DataDistributor"""

    def setUp(self) -> None:
        self.instance = factories.DataDistributorFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)


class TestDataDistributorLedger(TestCase):
    """Test cases for model DataDistributorLedger"""

    def setUp(self) -> None:
        self.instance = factories.DataDistributorLedgerFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.id))


class TestMarker(TestCase):
    """Test cases for model Marker"""

    fixtures = ["insights/tests/fixtures/Marker.json"]

    def setUp(self) -> None:
        self.instance = insight_models.Marker.objects.first()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)


class TestDistributorMarker(TestCase):
    """Test cases for model DistributorMarker"""

    def setUp(self) -> None:
        self.instance = factories.DistributorMarkerFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)


class TestDistributorMarkerPair(TestCase):
    """Test cases for model DistributorMarkerPair"""

    def setUp(self) -> None:
        self.instance = factories.DistributorMarkerPairFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)

    def test_get_value_function(self):
        valid_functions = [operator.add, operator.sub, operator.mul, operator.truediv]
        self.assertIn(self.instance.get_value_function(), valid_functions)

    def test_get_health_insight(self):
        health_insight = self.instance.marker.health_insights.first()
        other_health_insight = self.instance.marker.health_insights.last()
        # No combination supplied
        self.assertRaises(ValueError, self.instance.get_health_insight)
        # At most only one health insight should be returned per combination
        self.assertEqual(health_insight, other_health_insight)

    def test_create_patient_insights(self):
        patient_insight = factories.PatientHealthInsightFactory()
        derived_insights = self.instance.create_patient_insights(
            patient_insight.patient.id, patient_insight.collection_date
        )
        self.assertIsInstance(derived_insights, list)


class TestHealthInsight(TestCase):
    """Test cases for model HealthInsight"""

    fixtures = [
        "insights/tests/fixtures/Statistic.json",
        "insights/tests/fixtures/Marker.json",
        "insights/tests/fixtures/HealthInsight.json",
    ]

    def setUp(self) -> None:
        self.instance = insight_models.HealthInsight.objects.first()
        return super().setUp()

    def test_str_representation(self):
        obj = self.instance
        expected = f"{obj.id} - {obj.marker.name} - {obj.statistic.name}"
        self.assertEqual(str(obj), str(expected))


class TestHealthInsightCategory(TestCase):
    """Test cases for model HealthInsightCategory"""

    def setUp(self) -> None:
        self.instance = factories.HealthInsightCategoryFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.title))

    def test_instance_attrs(self):
        attrs = ["title", "slug"]
        for attr in attrs:
            self.assertTrue(hasattr(self.instance, attr))
            self.assertIsNotNone(getattr(self.instance, attr))


class TestHealthInsightStateValue(TestCase):
    """Test cases for model HealthInsightStateValue"""

    CONDITIONS = factories.HealthInsightStateValueOptions.CONDITIONS
    NULL_CONDITION = factories.HealthInsightStateValueOptions.NULL_CONDITION

    def setUp(self) -> None:
        self.instance = factories.HealthInsightStateValueFactory()
        return super().setUp()

    def test_instance_state_and_condition(self):
        """Test that instance's state complies with conditions"""
        obj = self.instance
        marker = obj.health_insight.marker
        self.assertTrue(marker.name in self.CONDITIONS)
        self.assertTrue(obj.state in factories.HealthInsightStateValueOptions.STATES)
        marker_condition = factories.HealthInsightStateValueOptions.get_condition(
            marker.name, obj.state, obj.applies_to_gender
        )
        if marker_condition:
            lower_limit = marker_condition.get("lower_limit")
            upper_limit = marker_condition.get("upper_limit")
            if lower_limit:
                self.assertTrue(obj.lower_limit >= lower_limit)
            if upper_limit:
                self.assertTrue(obj.upper_limit <= upper_limit)


class TestHealthInsightRecommendation(TestCase):
    """Test cases for model HealthInsightRecommendation"""

    fixtures = [
        "insights/tests/fixtures/Statistic.json",
        "insights/tests/fixtures/Marker.json",
        "insights/tests/fixtures/HealthInsight.json",
        "insights/tests/fixtures/HealthInsightRecommendation.json",
    ]

    def setUp(self) -> None:
        self.instance = insight_models.HealthInsightRecommendation.objects.first()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.id))

    def test_instance_attrs(self):
        attrs = ["title", "slug", "status"]
        for attr in attrs:
            self.assertTrue(hasattr(self.instance, attr))
            self.assertIsNotNone(getattr(self.instance, attr))


class TestPatientHealthInsightRecommendation(TestCase):
    """Test cases for model PatientHealthInsightRecommendation"""

    def setUp(self) -> None:
        self.instance = factories.PatientHealthInsightRecommendationFactory()
        return super().setUp()

    def test_instance_attrs(self):
        attrs = ["status", "patient", "recommendation"]
        for attr in attrs:
            self.assertTrue(hasattr(self.instance, attr))
            self.assertIsNotNone(getattr(self.instance, attr))

    def test_get_for_patient(self):
        patient = self.instance.patient
        qs = insight_models.PatientHealthInsightRecommendation.get_for_patient(patient)
        self.assertTrue(qs.filter(id=self.instance.id).exists())


class TestDeviceSupportedHealthInsight(TestCase):
    """Test cases for model DeviceSupportedHealthInsight"""

    def setUp(self) -> None:
        self.instance = factories.DeviceSupportedHealthInsightFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.id))


class TestPatientHealthInsight(TestCase):
    """Test cases for model PatientHealthInsight"""

    def setUp(self) -> None:
        self.instance = factories.PatientHealthInsightFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.id))

    def test_clean_period_avg_interval(self) -> None:
        ref_date = self.instance.collection_date
        intervals = {
            "day": {
                "good": {"days": [6, 15, 29], "weeks": [1, 2, 4]},
                "bad": {"days": [32, 60, 90, 365], "weeks": [5, 8, 53]},
            },
            "month": {
                "good": {"days": [15, 30, 60, 180, 330], "weeks": [1, 2, 4, 8, 16, 52]},
                "bad": {"days": [365, 548], "weeks": [53, 104]},
            },
            "quarter": {
                "good": {"days": [15, 45, 90, 180, 330], "weeks": [1, 4, 8, 16, 52]},
                "bad": {"days": [365, 548], "weeks": [53, 104]},
            }
            # any range is valid in "year" and "week_day" period aggregation
        }
        try:
            for p, interval in intervals.items():
                for day in interval["good"]["days"]:
                    start = ref_date - timedelta(days=day)
                    res = self.instance.clean_period_avg_interval(p, start, ref_date)
                    self.assertIsNone(res)
                for week in interval["good"]["weeks"]:
                    start = ref_date - timedelta(weeks=week)
                    res = self.instance.clean_period_avg_interval(p, start, ref_date)
                    self.assertIsNone(res)
                for day in interval["bad"]["days"]:
                    start = ref_date - timedelta(days=day)
                    with self.assertRaises(ValueError):
                        self.instance.clean_period_avg_interval(p, start, ref_date)
                for week in interval["bad"]["weeks"]:
                    start = ref_date - timedelta(weeks=week)
                    with self.assertRaises(ValueError):
                        self.instance.clean_period_avg_interval(p, start, ref_date)
        except Exception as e:
            self.fail(f"clean_period_avg_interval({p},{start},{ref_date}) Raised: {e}")

    def test_period_avg(self) -> None:  # sourcery skip: low-code-quality
        """Test period_avg method of PatientHealthInsight model
        This method is used to calculate the average value of a health insight
        in a period of time. This test checks against a sample of periods if
        the method returns the correct value.
        """
        ref_date = datetime.combine(
            self.instance.collection_date, time.min, tzinfo=pytz.utc
        )
        periods = bad_periods = {p: [] for p in self.instance.PERIOD_AGG.keys()}
        for i in range(
            1, 6
        ):  # test date ranges from 1 to 6 weeks/months/quarters/years
            periods["week"].append((ref_date - timedelta(weeks=i), ref_date))
            periods["month"].append((ref_date - relativedelta(months=i), ref_date))
            periods["quarter"].append(
                (ref_date - relativedelta(months=3 * i), ref_date)
            )

        # for each testing period, get the avg of the values for this insight instance
        for start, end in list(
            chain.from_iterable(periods.values())
        ):  # all date ranges
            for p in self.instance.PERIOD_AGG:  # test all period aggregation options
                try:
                    self.instance.clean_period_avg_interval(p, start, end)
                except ValueError:
                    bad_periods[p].append((p, start, end))
                    # bad_periods are ranges not valid for agg type.
                    # eg. day aggregation in >=1 month range
                    continue  # skip to next period aggregation type
                insight = self.instance.health_insight
                try:
                    period_avg = self.instance.period_avg(
                        self.instance.patient, insight, start, end, period=p
                    )
                    self.assertIsInstance(period_avg, models.QuerySet)  # return type
                    # Test row count upper limit according to period aggregation
                    row_count = period_avg.count()
                    diff_days = (end - start).days  # adds 2 to include start and end
                    if p == "week":  # at most 53 week values
                        self.assertLessEqual(row_count, 53)
                        self.assertLessEqual(row_count, diff_days // 7 + 2)
                    if p == "week_day":
                        self.assertLessEqual(row_count, 7)
                        self.assertLessEqual(row_count, diff_days + 2)
                    if p == "day":
                        self.assertLessEqual(row_count, 31)
                        self.assertLessEqual(row_count, diff_days + 2)
                    if p == "month":
                        self.assertLessEqual(row_count, 12)
                        self.assertLessEqual(row_count, diff_days // 30 + 2)
                    if p == "quarter":
                        self.assertLessEqual(row_count, 4)
                        self.assertLessEqual(row_count, diff_days // 90 + 2)
                    if p == "year":
                        self.assertLessEqual(row_count, diff_days // 365 + 2)

                    # Test correctness of average value
                    for row in period_avg:
                        qs = insight_models.PatientHealthInsight.objects.filter(
                            collection_date__range=(start, end),
                            health_insight=insight,
                        ).values(
                            **{
                                p: self.instance.PERIOD_AGG[p](
                                    "collection_date", tzinfo=pytz.utc
                                )
                            }
                        )
                        query_value = qs.aggregate(avg=models.Avg("value"))["avg"]
                        self.assertEqual(row["avg"], query_value)
                except Exception as e:
                    self.fail(f"period_avg({insight},{start},{end},{p}) Raised: {e}")

    def test_state(self) -> None:
        """Test state property of PatientHealthInsight model
        This property is used to get the state of the insight value instance.
        This test checks if the state is correctly retrieved.
        """
        STATES = insight_models.HealthInsightStateValue.States
        self.assertTrue(self.instance.state.get("name") in STATES)


class TestHealthInsightDisplayRuleSet(TestCase):
    """Test cases for model HealthInsightDisplayRuleSet"""

    def setUp(self) -> None:
        self.instance = factories.HealthInsightDisplayRuleSetFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), self.instance.name)

    def test_save_ruleset(self) -> None:
        """Test save method of HealthInsightDisplayRuleSet model"""
        # Test if rule is correctly saved with an existing rule set
        rule_set = factories.HealthInsightDisplayRuleSetFactory.create()
        rule_set.save()
        result = factories.HealthInsightDisplayRuleFactory.create(
            rule_set=self.instance
        )
        result.save()
        self.assertIsNotNone(result.id)
        self.assertIsNotNone(result.rule_set.id)


class TestHealthInsightDisplayRule(TestCase):
    """Test cases for model HealthInsightDisplayRule"""

    def setUp(self) -> None:
        self.instance = factories.HealthInsightDisplayRuleFactory()
        self.patient_insight = factories.PatientHealthInsightFactory()
        return super().setUp()

    def test_str_representation(self):
        self.assertEqual(str(self.instance), str(self.instance.id))

    def test_eval(self):
        """Test eval method of HealthInsightDisplayRule model
        This method is used to evaluate the rule against a patient insight.
        This test checks if the rule is correctly evaluated by Python eval.
        """
        # Test if rule is correctly evaluated
        value = self.instance.get_field_value(self.patient_insight)
        expect = eval(
            f"{value} {self.instance.comparator} {self.instance.comparison_value}"
        )
        self.assertEqual(self.instance.eval(self.patient_insight), expect)

    def test_get_expression(self):
        """Test get_expression method of HealthInsightDisplayRule model
        This method is used to get the Python expression of the rule.
        This test checks if the expression is correctly retrieved.
        """
        # Test if expression is correctly retrieved
        value = self.instance.get_field_value(self.patient_insight)
        expect = f"{value} {self.instance.comparator} {self.instance.comparison_value}"
        self.assertEqual(self.instance.get_expression(self.patient_insight), expect)
        # Test if expression is correctly retrieved for collection_date field
        self.assertEqual(
            self.instance.get_expression(self.patient_insight),
            f"{value} {self.instance.comparator} {self.instance.comparison_value}",
        )

    def test_get_field_value(self):
        """Test get_field_value method of HealthInsightDisplayRule model
        This method is used to get the query value of the field for a rule.
        This test checks if the value is correctly retrieved.
        """
        # Test if value is correctly retrieved for its field
        if self.instance.field == "value":
            self.assertEqual(
                self.instance.get_field_value(self.patient_insight),
                self.patient_insight.value,
            )
        elif self.instance.field == "collection_date":
            days_ago = int(self.instance.comparison_value)
            self.assertLessEqual(
                days_ago, self.instance.get_field_value(self.patient_insight)
            )
