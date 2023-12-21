import unittest
import json

from app.api import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_get_client_data(self):
        response = self.app.get('/client_data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('client_data', data)

    def test_get_list_features(self):
        response = self.app.get('/get_list_features')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('list_features', data)

    def test_predict(self):
        data = {
            'features': {
                'SK_ID_CURR': 100167
            }
        }
        response = self.app.get('/predict', json=data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)

    def test_get_shap(self):
        data = {
            'features': {
                'SK_ID_CURR': 100167
            }
        }
        response = self.app.get('/get_shap', json=data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('shap_values', data)

if __name__ == '__main__':
    app.load_client_data()
    unittest.main()
