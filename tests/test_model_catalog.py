import unittest

from src.processing.model_catalog import diarization_catalog, transcription_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_transcription_catalog_matches_available_runtime_models(self):
        models = transcription_catalog()
        self.assertEqual([item.key for item in models], [
            "large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"
        ])
        self.assertTrue(all("/" in item.repo for item in models))

    def test_diarization_catalog_exposes_active_pyannote_model(self):
        models = diarization_catalog()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].repo, "pyannote/speaker-diarization-community-1")


if __name__ == "__main__":
    unittest.main()
