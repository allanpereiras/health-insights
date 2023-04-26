import logging
from contextlib import suppress
from datetime import date, timedelta

from django.conf import settings
from django.contrib.auth.models import User
from django.urls import reverse
from insights.api.utils import RedshiftUnavailable
from insights.models import HealthInsight
from insights.tests import factories
from patients.tests import factories as patient_factories
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

logger = logging.getLogger("django.request")
previous_level = logger.getEffectiveLevel()
logger.setLevel(logging.ERROR)


class BaseTestCase(APITestCase):
    """Base class for test cases in this module"""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="foo")
        self.client.force_authenticate(user=self.user)

    def assert_get_response(self, url, status):
        """Assert expected response status for a GET request"""
        response = self.client.get(url, format="json")
        self.assertEqual(response.status_code, status)
        return response


class TestDataDistributorLedgerViewSet(BaseTestCase):
    """Test cases for DataDistributorLedgerViewSet endpoints"""

    def setUp(self):
        """Set instances and other resources needed for testing"""
        super().setUp()
        self.instance = factories.DataDistributorLedgerFactory()
        self.external_patient = self.instance.patient.external_patients.first()
        if not self.external_patient:
            self.external_patient = patient_factories.ExternalPatientFactory()

    def test_get_ledger_batch(self):
        """Ensure we can get a ledger batch the same size as set in env var"""
        params = {  # URL Params in regex
            "ext_patient_id": self.external_patient.external_id,
            "distrib_name": self.instance.distributor.name,
        }
        url = reverse("insights:datadistributorledger-get-ledger-batch", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_200_OK)
        batch_size = int(settings.LEDGERS_BATCH_SIZE)
        self.assertLessEqual(response.data["count"], batch_size)

    def test_get_smaller_ledger_batch(self):
        """Distributor's base_date lower limit for ledgers start date"""
        # A recent DataDistributor base_date will give a smaller batch size (<30 days)
        self.instance.distributor.base_date = date.today() - timedelta(days=7)
        self.instance.distributor.save()
        params = {  # URL Params in regex
            "ext_patient_id": self.external_patient.external_id,
            "distrib_name": self.instance.distributor.name,
        }
        url = reverse("insights:datadistributorledger-get-ledger-batch", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_200_OK)
        batch_size = int(settings.LEDGERS_BATCH_SIZE)
        self.assertLessEqual(response.data["count"], batch_size)

    def test_get_ledger_batch_invalid_distributor(self):
        params = {  # URL Params in regex
            "ext_patient_id": self.external_patient.external_id,
            "distrib_name": "INVALID_NAME",
        }
        url = reverse("insights:datadistributorledger-get-ledger-batch", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_404_NOT_FOUND)
        self.assertNotIn("count", response.data)  # No obj count key in response data


class TestPatientHealthInsightViewSet(BaseTestCase):
    """Test cases for PatientHealthInsightViewSet endpoints"""


    def setUp(self):
        """Set instances and other resources needed for testing"""
        super().setUp()
        self.instance = factories.PatientHealthInsightFactory()
        self.external_patient = self.instance.patient.external_patients.first()
        if not self.external_patient:
            self.external_patient = patient_factories.ExternalPatientFactory()

    def test_redshift_values(self):
        """Test response for redshift_values endpoint listing all calculated
        Health Insights for a Patient, Distributor and collection date
        """
        params = {  # URL Params in regex
            "ext_patient_id": self.external_patient.external_id,
            "distrib": self.instance.data_distributor.name,
            "collection_date": self.instance.collection_date,
        }  # since instance is random from factory, we can use its values
        url = reverse("insights:patienthealthinsight-redshift-values", kwargs=params)
        with suppress(RedshiftUnavailable):
            response = self.client.get(url, format="json")
            valid_status = [status.HTTP_200_OK, RedshiftUnavailable.status_code]
            self.assertIn(response.status_code, valid_status)

    def test_redshift_values_invalid(self):
        """Test invalid response for redshift_values endpoint with invalid params"""
        params = {  # URL Params in regex
            "ext_patient_id": self.external_patient.external_id,
            "distrib": self.instance.data_distributor.name,
            "collection_date": date.today() + timedelta(days=1),  # future date
        }
        url = reverse("insights:patienthealthinsight-redshift-values", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_400_BAD_REQUEST)
        self.assertNotIn("count", response.data)  # No obj count key in response data
        params["collection_date"] = "1012-13-99"  # invalid date format
        url = reverse("insights:patienthealthinsight-redshift-values", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_400_BAD_REQUEST)
        self.assertNotIn("count", response.data)
        params["collection_date"] = date.today()  # valid date format
        params["distrib"] = "INVALID_NAME"  # invalid distributor name
        url = reverse("insights:patienthealthinsight-redshift-values", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_404_NOT_FOUND)
        self.assertNotIn("count", response.data)

    def test_display(self):
        """Test response for display endpoint listing Health Insights for
        display according to HealthInsightDisplayRuleSet
        """
        params = {"ext_patient_id": self.external_patient.external_id}
        url = reverse("insights:patienthealthinsight-display", kwargs=params)
        self.assert_get_response(url, status.HTTP_200_OK)
    
    def test_latest(self):
        """Test response for latest endpoint listing latest Health Insights for
        an ExternalPatient ID
        """
        params = {"ext_patient_id": self.external_patient.external_id}
        url = reverse("insights:patienthealthinsight-latest", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_200_OK)
        self.assertLessEqual(response.data["count"], HealthInsight.objects.count())

    def test_chart(self):
        """Test response for chart endpoint that returns average value of a
        Patient Health Insight for a period
        """
        params = {
            "ext_patient_id": self.external_patient.external_id,
            "h_insight_id": self.instance.health_insight.id,
        }
        url = reverse("insights:patienthealthinsight-chart", kwargs=params)
        response = self.assert_get_response(url, status.HTTP_200_OK)
        attrs = [
            "chart_data",
            "chart_data_statistics",
            "health_insight",
            "from_date",
            "to_date",
            "period",
        ]
        for attr in attrs:
            self.assertIn(attr, response.data)

    def test_chart_invalid(self):
        """Test invalid response for chart endpoint with invalid params"""
        params = {
            "ext_patient_id": self.external_patient.external_id,
            "h_insight_id": 0,  # invalid health insight id
        }
        url = reverse("insights:patienthealthinsight-chart", kwargs=params)
        self.assert_get_response(url, status.HTTP_404_NOT_FOUND)


class TestPatientRecommendationsViewSet(BaseTestCase):
    """Test cases for PatientRecommendationsViewSet endpoints"""

    def setUp(self):
        """Set instances and other resources needed for testing"""
        super().setUp()
        self.instance = factories.PatientHealthInsightRecommendationFactory()
        self.external_patient = self.instance.patient.external_patients.first()

    def test_get_recommendations_for_patient(self):
        """Test response for get_recommendations_for_patient endpoint listing
        Recommendations generated from Health Insights of a Patient
        """
        params = {"ext_patient_id": self.external_patient.external_id}
        url = reverse(
            "insights:patients-recommendations-get-recommendations-for-patient",
            kwargs=params,
        )
        self.assert_get_response(url, status.HTTP_200_OK)
