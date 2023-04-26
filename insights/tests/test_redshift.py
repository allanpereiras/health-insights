import random

import pytz
from django.db.models import QuerySet
from django.test import TestCase
from faker import Faker
from insights import redshift
from insights.models import PatientHealthInsight
from insights.tests import factories
from patients.tests.factories import ExternalPatientFactory
from pytz import UTC

faker = Faker()


class RedshiftSetUpTestCase(TestCase):

    def setUp(self) -> None: 
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()


class PatientInsightsTestCase(RedshiftSetUpTestCase):

    def setUp(self) -> None:
        self.external_patient = ExternalPatientFactory()
        self.distributor = factories.DataDistributorFactory()
        self.collection_date = faker.date_time_between(
            start_date="-1w", end_date="now", tzinfo=UTC
        ) # From the past week
        self.tz = pytz.timezone("UTC")
        # Fabricate Health Insight's related objects for usage in the test functions
        statistics = [
            factories.StatisticFactory(name=s) for s in ["avg", "max", "min", "sum"]
        ]
        markers = [
            factories.MarkerFactory(name=n) for n in factories.MarkerOptions.ALL
        ]
        distributor_markers = [
            factories.DistributorMarkerFactory(marker=m) for m in markers
        ]
        health_insight_list = [
            factories.HealthInsightFactory(marker=m.marker) for m in distributor_markers
        ]
        self.patient_insights = [
            factories.PatientHealthInsightFactory(
                patient=self.external_patient.patient,
                data_distributor=self.distributor, # Fixed distributor
                health_insight=health_insight,
                collection_date=self.collection_date,
            ) for health_insight in health_insight_list
        ]
        self.instance = redshift.PatientInsights(
            self.external_patient.external_id, self.collection_date.date(), str(self.tz)
        )

    def test_bad_collection_date(self) -> None:
        future_date = faker.future_date(tzinfo=UTC)
        self.assertRaises(
            ValueError,
            redshift.PatientInsights,
            self.external_patient.external_id,
            future_date,
            "UTC",
        )
        wrong_format = "1 Jan 01"
        self.assertRaises(
            ValueError,
            redshift.PatientInsights,
            self.external_patient.external_id,
            wrong_format,
            "UTC",
        )

    def test_get_period_for_insight(self) -> None:
        for insight in self.instance.health_insights:
            period = self.instance.get_period_for_insight(insight.health_insight)
            self.assertIsInstance(period, tuple)  # (start_date, end_date)
            self.assertEqual(len(period), 2)
            self.assertEqual(
                period, insight.health_insight.get_period(self.collection_date, self.tz)
            )

    def test_get_or_create_insight(self) -> None:
        insight = random.choice(self.instance.health_insights)
        distributor = insight.data_distributor
        obj, created = self.instance.get_or_create_insight(
            insight.health_insight, distributor
        )
        self.assertIsInstance(obj, PatientHealthInsight)
        self.assertFalse(created)  # Insight already exists from random.choice
    
    def test_get_or_create_all_insights(self) -> None:
        self.instance.get_or_create_all_insights(self.distributor.name)
        insight_list = self.instance.health_insights
        self.assertIsInstance(insight_list, QuerySet)
        self.assertTrue(
            all(isinstance(insight, PatientHealthInsight) for insight in insight_list)
        )

    def test_process_options(self) -> None:
        self.instance.get_or_create_all_insights(self.distributor.name)
        insight = random.choice(self.instance.health_insights)
        available_options = {
            "sql": self.instance.process_with_sql,
            "python": self.instance.process_with_python,
        }
        for process in available_options.values():
            query_result = process(insight)
            self.assertIsInstance(query_result, list)  # list of tuples/dicts
