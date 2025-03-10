# reasoning/ch02.py

class LLM:
    def __init__(self, model_name="placeholder-model"):
        self.model_name = model_name

    def predict(self, prompt):
        """
        Placeholder function that returns a canned response.
        Replace with your actual language model logic.
        """
        return f"Model ({self.model_name}) response to: '{prompt}'"


def generate(model, prompt):
    """
    Placeholder function that calls the model's predict method.
    Replace with your actual generation logic.
    """
    return model.predict(prompt)