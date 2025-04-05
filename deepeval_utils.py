from deepeval.models import DeepEvalBaseLLM


class DeepevalGeminiModel(DeepEvalBaseLLM):
    def __init__(self, client, model_name: str = "gemini-2.0-flash-001"):
        self.model_name = model_name
        self.client = client

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        # model_name = self.load_model()
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
