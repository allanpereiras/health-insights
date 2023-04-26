"""
This module provide classes for mocking instances of models from Insights App.

It uses Factory Boy python package in combination with Faker for building
objects that can be used for testing purposes.
"""

import random
from datetime import date, datetime, timedelta

import factory
from django.utils import timezone
from factory.fuzzy import FuzzyChoice, FuzzyFloat
from faker import Faker
from insights import models
from patients.tests.factories import PatientFactory
from pytz import utc


class DataDeviceFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Data Devices"""

    name = factory.Faker(
        "random_element",
        elements=(
            "Apple Watch",
            "Oura Ring",
            "Garmin Venu",
            "Garmin Fenix",
            "Fitbit Versa",
            "Fitbit Charge",
        ),
    )
    version = factory.Faker("numerify", text="v#.#.#")

    class Meta:
        model = "insights.DataDevice"
        django_get_or_create = ("name", "version")


class StatisticFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Statistics"""

    name = factory.Iterator(["sum", "avg", "min", "max"])
    slug_name = factory.LazyAttribute(lambda o: o.name)

    class Meta:
        model = "insights.Statistic"
        django_get_or_create = ("name",)


class PatientDeviceFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Patient Device"""

    patient = factory.SubFactory(PatientFactory)
    device = factory.SubFactory(DataDeviceFactory)

    class Meta:
        model = "insights.PatientDevice"
        django_get_or_create = ("patient", "device")


class DataDistributorFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Data Distributors"""

    name = factory.Iterator(["APPLE_HEALTH", "GOOGLE_FIT"])

    @factory.lazy_attribute
    def base_date(self):
        """Random base_date from 90 up to 30 days ago"""
        start_date = date.today() - timedelta(days=90)
        end_date = date.today() - timedelta(days=30)
        return Faker().date_between(start_date, end_date)

    class Meta:
        model = "insights.DataDistributor"
        django_get_or_create = ("name",)


class DataDistributorLedgerFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Data Distributor Ledgers"""

    patient = factory.SubFactory(PatientFactory)
    distributor = factory.SubFactory(DataDistributorFactory)
    processed_at = factory.Maybe(
        "is_processed",
        yes_declaration=factory.Faker("past_datetime", tzinfo=utc),
        no_declaration=None,
    )
    file_location = factory.Maybe(
        "is_processed",
        yes_declaration=factory.Faker("file_path", depth=5),
        no_declaration=None,
    )

    @factory.lazy_attribute
    def started_at(self):
        start_dt = datetime.combine(self.distributor.base_date, datetime.min.time())
        return timezone.make_aware(start_dt)

    @factory.lazy_attribute
    def finished_at(self):
        finished_dt = Faker().past_datetime(start_date=self.started_at)
        return timezone.make_aware(finished_dt)

    class Meta:
        model = "insights.DataDistributorLedger"


class MarkerOptions:
    CALORIES = "Calories Burned"
    HEALTH_SCORE = "Health Score"
    RESTING_HEART_RATE = "Resting Heart Rate"
    SLEEP = "Sleep Duration"
    STEPS = "Steps"
    WEIGHT = "Weight"

    ALL = [CALORIES, HEALTH_SCORE, RESTING_HEART_RATE, SLEEP, STEPS, WEIGHT]


class MarkerFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Markers"""

    name = factory.Iterator(MarkerOptions.ALL)

    class Meta:
        model = "insights.Marker"
        django_get_or_create = ("name",)


class DistributorMarkerFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Distributor Markers"""

    marker = factory.SubFactory(MarkerFactory)
    distributor = factory.SubFactory(DataDistributorFactory)
    name = factory.LazyAttribute(lambda o: f"{o.distributor.name} - {o.marker.name}")

    class Meta:
        model = "insights.DistributorMarker"


class DistributorMarkerPairFactory(DistributorMarkerFactory):
    """Mock Factory for Distributor Markers"""

    first = factory.SubFactory(DistributorMarkerFactory)
    value_function = FuzzyChoice(
        models.DistributorMarkerPair.OPERATIONS, getter=lambda c: c[0]
    )
    second = factory.SubFactory(DistributorMarkerFactory)
    units = factory.Iterator(
        ["lb", "steps", "calories", "bpm", "hr", "kcal", "count/min", "count"]
    )

    class Meta:
        model = "insights.DistributorMarkerPair"


class HealthInsightFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Health Insights"""

    marker = factory.SubFactory(MarkerFactory)
    statistic = factory.SubFactory(StatisticFactory)

    class Meta:
        model = "insights.HealthInsight"
        django_get_or_create = ("marker", "statistic")


class HealthInsightCategoryFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Health Insight Categories"""

    title = factory.Iterator([f"Category {i}" for i in range(1, 6)])

    class Meta:
        model = "insights.HealthInsightCategory"

    @factory.post_generation
    def health_insights(self, create, extracted, **kwargs):
        if not create:
            # Simple build, do nothing.
            return
        if extracted:
            # A list of health_insights was passed in, use it
            for insight in extracted:
                self.health_insights.add(insight)


class HealthInsightStateValueOptions:
    STATES = models.HealthInsightStateValue.States
    NULL_CONDITION = {"state": STATES.UNKNOWN, "lower_limit": 0, "upper_limit": 0}
    CONDITIONS = {
        MarkerOptions.HEALTH_SCORE: [
            {"state": STATES.RISK, "lower_limit": None, "upper_limit": 69},
            {"state": STATES.WARNING, "lower_limit": 70, "upper_limit": 78},
            {"state": STATES.SUCCESS, "lower_limit": 79, "upper_limit": None},
        ],
        MarkerOptions.STEPS: [
            {"state": STATES.WARNING, "lower_limit": None, "upper_limit": 9999},
            {"state": STATES.SUCCESS, "lower_limit": 10000, "upper_limit": None},
        ],
        MarkerOptions.RESTING_HEART_RATE: [
            {
                "state": STATES.WARNING,
                "lower_limit": None,
                "upper_limit": 60,
                "applies_to_gender": "Male",
                "reversed_cumulative": True,
            },
            {
                "state": STATES.WARNING,
                "lower_limit": None,
                "upper_limit": 80,
                "applies_to_gender": "Female",
                "reversed_cumulative": True,
            },
            {
                "state": STATES.SUCCESS,
                "lower_limit": 61,
                "upper_limit": None,
                "applies_to_gender": "Male",
                "reversed_cumulative": True,
            },
            {
                "state": STATES.SUCCESS,
                "lower_limit": 81,
                "upper_limit": None,
                "applies_to_gender": "Female",
                "reversed_cumulative": True,
            },
        ],
        MarkerOptions.SLEEP: [
            {"state": STATES.WARNING, "lower_limit": None, "upper_limit": 6},
            {"state": STATES.SUCCESS, "lower_limit": 7, "upper_limit": None},
        ],
        MarkerOptions.CALORIES: [
            {"state": STATES.WARNING, "lower_limit": None, "upper_limit": 499},
            {"state": STATES.SUCCESS, "lower_limit": 500, "upper_limit": None},
        ],
        MarkerOptions.WEIGHT: [NULL_CONDITION],
    }

    @classmethod
    def get_condition(cls, marker: str, state: str, gender: str) -> dict:
        """Get the condition for a given marker and state"""
        marker_conditions = list(
            filter(lambda d: d["state"] == state, cls.CONDITIONS.get(marker))
        )
        if len(marker_conditions) > 1:  # Resting HR - Male/Female conditions
            marker_conditions = list(
                filter(
                    lambda d: d["applies_to_gender"] == gender,
                    marker_conditions,
                )
            )
        return marker_conditions[0] if marker_conditions else None


class HealthInsightStateValueFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Health Insight State Values"""

    health_insight = factory.SubFactory(HealthInsightFactory)
    state = FuzzyChoice(
        models.HealthInsightStateValue.States.choices, getter=lambda c: c[0]
    )
    applies_to_gender = FuzzyChoice(
        models.HealthInsightStateValue.Genders.choices, getter=lambda c: c[0]
    )

    class Meta:
        model = "insights.HealthInsightStateValue"
        django_get_or_create = ("health_insight",)

    @factory.lazy_attribute
    def upper_limit(self):
        marker = self.health_insight.marker.name
        gender = self.applies_to_gender
        opt = HealthInsightStateValueOptions.get_condition(marker, self.state, gender)
        if opt:
            lower_limit = opt.get("lower_limit", 0) or 0
            upper_limit = opt.get("upper_limit", 10000) or 10000
        else:
            lower_limit = 0
            upper_limit = 10000
        for_range = [lower_limit, upper_limit]
        return FuzzyFloat(min(*for_range), max(*for_range)).fuzz()

    @factory.lazy_attribute
    def lower_limit(self):
        marker = self.health_insight.marker.name
        gender = self.applies_to_gender
        opt = HealthInsightStateValueOptions.get_condition(marker, self.state, gender)
        lower_limit = opt.get("lower_limit", 0) or 0 if opt else 0
        return FuzzyFloat(lower_limit, self.upper_limit).fuzz()


class DeviceSupportedHealthInsightFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Device Supported Health Insights"""

    device = factory.SubFactory(DataDeviceFactory)
    health_insight = factory.SubFactory(HealthInsightFactory)

    class Meta:
        model = "insights.DeviceSupportedHealthInsight"
        django_get_or_create = ("device", "health_insight")


class PatientHealthInsightFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Patient Health Insights"""

    health_insight = factory.SubFactory(HealthInsightFactory)
    data_distributor = factory.SubFactory(DataDistributorFactory)
    patient = factory.SubFactory(PatientFactory)
    collection_date = factory.Faker("past_date", tzinfo=utc)
    value = factory.Faker("pyfloat", left_digits=3, right_digits=2, positive=True)
    last_collection_time = factory.Faker("time_object")

    class Meta:
        model = "insights.PatientHealthInsight"

    @factory.lazy_attribute
    def units(self):
        UNITS = ["lb", "steps", "calories", "bpm", "hr", "kcal", "count/min", "count"]
        return random.choice(UNITS)


class GoalFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Goals"""

    aggregation_type = FuzzyChoice(
        models.Goal.Aggregations.choices, getter=lambda c: c[0]
    )

    class Meta:
        model = "insights.Goal"


class HealthInsightRecommendationFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Health Insight Recommendations"""

    goal = factory.SubFactory(GoalFactory)
    title = factory.Iterator([f"Recommendation {i}" for i in range(1, 6)])

    class Meta:
        model = "insights.HealthInsightRecommendation"

    @factory.post_generation
    def health_insights(self, create, extracted, **kwargs):
        self.health_insights.add(HealthInsightFactory())


class PatientHealthInsightRecommendationFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Patient Health Insight Recommendations"""

    patient = factory.SubFactory(PatientFactory)
    recommendation = factory.SubFactory(HealthInsightRecommendationFactory)

    class Meta:
        model = "insights.PatientHealthInsightRecommendation"


class HealthInsightDisplayRuleSetFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Health Insight Display Rule Set"""

    health_insight = factory.SubFactory(HealthInsightFactory)
    priority = factory.Faker("random_digit_not_null")
    name = factory.Iterator(
        [
            "Latest",
            "Latest overnight HR score",
            "Last night's only",
            "Today's total",
        ]
    )

    class Meta:
        model = "insights.HealthInsightDisplayRuleSet"


class HealthInsightDisplayRuleFactory(factory.django.DjangoModelFactory):
    """Mock Factory for Health Insight Display Rule (Comparison Rule)"""

    rule_set = factory.SubFactory(HealthInsightDisplayRuleSetFactory)
    field = FuzzyChoice(
        models.HealthInsightDisplayRule.PatientHealthInsightField.choices,
        getter=lambda c: c[0],
    )
    comparator = FuzzyChoice(
        models.HealthInsightDisplayRule.Comparator.choices, getter=lambda c: c[0]
    )
    comparison_value = factory.Faker("numerify", text="#")
    comparison_value_type = FuzzyChoice(
        models.HealthInsightDisplayRule.ComparisonValueType.choices,
        getter=lambda c: c[0],
    )

    class Meta:
        model = "insights.HealthInsightDisplayRule"

    @factory.lazy_attribute
    def comparison_value_type(self):
        collec_date = (
            models.HealthInsightDisplayRule.PatientHealthInsightField.COLLECTION_DATE
        )
        if self.field == collec_date:
            return models.HealthInsightDisplayRule.ComparisonValueType.DAYS_AGO
        return models.HealthInsightDisplayRule.ComparisonValueType.DECIMAL


class RedshiftConnection:
    """Mock Redshift Connection"""

    COLUMN_NAMES = {
        "ledger_uuid",
        "external_patient_id",
        "distributor",
        "distributor_marker",
        "device_name",
        "device_version",
        "value_timestamp",
        "value_units",
        "distributor_version",
        "value",
        "created_at",
        "updated_at",
        "value_timestamp_start",
        "value_timestamp_end",
    }

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __str__(self) -> str:
        return dict(self)

    def retrieve_data(self, count=10, **kwargs):
        """Returns a list of DataFactory objects (rows from Redshift).
        Max: 200 objects per call (mimic Redshift limit)
        returns: list of DataFactory objects
        """
        return [DataFactory(**kwargs) for _ in range(count)]


class DataFactory(factory.Factory):
        """Mock Factory for Data in Redshift DB.
        Attributes must match column names in table.
        """

        class Meta:
            model = dict

        ledger_uuid = factory.Faker("uuid4")
        external_patient_id = factory.Faker("numerify", text="####")
        distributor = factory.Iterator(["APPLE_HEALTH", "GOOGLE_FIT"])
        distributor_marker = factory.Iterator(MarkerOptions.ALL)
        device_name = factory.Faker(
            "random_element",
            elements=(
                "Apple Watch",
                "Oura Ring",
                "Garmin Venu",
                "Garmin Fenix",
                "Fitbit Versa",
                "Fitbit Charge",
            ),
        )
        device_version = factory.Faker("numerify", text="v#.#.#")
        value_timestamp = factory.Faker("past_datetime", tzinfo=utc)
        value_units = factory.Iterator(
            ["lb", "steps", "calories", "bpm", "hr", "kcal", "count/min", "count"]
        )
        distributor_version = factory.Faker("numerify", text="v#.#.#")
        value = factory.Faker("pyfloat", left_digits=3, right_digits=2, positive=True)
        created_at = factory.Faker("past_datetime", tzinfo=utc)
        updated_at = factory.Faker("past_datetime", tzinfo=utc)
        value_timestamp_start = factory.Faker("past_datetime", tzinfo=utc)
        value_timestamp_end = factory.Faker("past_datetime", tzinfo=utc)


def get_seeding_list() -> list:
    """Returns a list of tuples for all Factory classes with its count for
    objects to create. Count is the length of Iterator names of each Factory.
    """
    return [
        (DataDeviceFactory, 1),
        (StatisticFactory, 4),
        (PatientDeviceFactory, 1),
        (DataDistributorFactory, 2),
        (DataDistributorLedgerFactory, 1),
        (MarkerFactory, 6),
        (DistributorMarkerFactory, 1),
        (DistributorMarkerPairFactory, 1),
        (HealthInsightFactory, 1),
        (DeviceSupportedHealthInsightFactory, 1),
        (PatientHealthInsightFactory, 1),
        (HealthInsightDisplayRuleSetFactory, 4),
        (HealthInsightDisplayRuleFactory, 1),
        (HealthInsightCategoryFactory, 1),
        (HealthInsightRecommendationFactory, 1),
        (HealthInsightStateValueFactory, 1),
    ]
