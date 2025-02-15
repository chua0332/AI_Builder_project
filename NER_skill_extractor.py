from transformers import pipeline
import warnings
import math
import re

class SkillKnowledgeExtractor:
    """
    Class to extract skills and knowledge using Hugging Face models.
    """

    def __init__(self):
        """Initialize the Hugging Face pipelines."""
        try:
            self.skill_pipeline = pipeline(
                model="jjzha/jobbert_skill_extraction", 
                aggregation_strategy="first"
            )
            self.knowledge_pipeline = pipeline(
                model="jjzha/jobbert_knowledge_extraction", 
                aggregation_strategy="first"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing pipelines: {e}")

    def _aggregate_span(self, results):
        """
        Merge consecutive spans for multi-word entities.
        """
        if not results:
            return []

        aggregated = []
        current = results[0]

        for result in results[1:]:
            if result["start"] == current["end"] + 1:
                current["word"] += " " + result["word"]
                current["end"] = result["end"]
            else:
                aggregated.append(current)
                current = result

        aggregated.append(current)
        return aggregated

    def extract_skills(self, text):
        """
        Extract skills from text using the skill extraction model.
        """
        if not isinstance(text, str):
            warnings.warn("Input should be a string.", UserWarning)
            return []

        output = self.skill_pipeline(text)
        for item in output:
            if "entity_group" in item:
                item["entity"] = "Skill"
                del item["entity_group"]
        
        return self._aggregate_span(output)

    def extract_knowledge(self, text):
        """
        Extract knowledge from text using the knowledge extraction model.
        """
        if not isinstance(text, str):
            warnings.warn("Input should be a string.", UserWarning)
            return []

        output = self.knowledge_pipeline(text)
        for item in output:
            if "entity_group" in item:
                item["entity"] = "Knowledge"
                del item["entity_group"]
        
        return self._aggregate_span(output)

    def extract_all(self, text):
        """
        Extract both skills and knowledge from the text.
        """
        skills = self.extract_skills(text)
        knowledge = self.extract_knowledge(text)
        return {
            "skills": [s["word"] for s in skills],
            "knowledge": [k["word"] for k in knowledge]
        }
        
    
