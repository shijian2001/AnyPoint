"""
Task embedding with three-component encoding.

Strategy: Encode layout, question, answer separately to same dimension D,
then concatenate: v_t = [v_layout || v_question || v_answer] ∈ R^(3D)
"""

from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from point_qa_generator.base import Task


class TaskEmbedder:
    """
    Three-component task encoder.
    
    Embedding:
        v_layout ∈ R^D
        v_question ∈ R^D
        v_answer ∈ R^D
        v_t = normalize([v_layout || v_question || v_answer]) ∈ R^(3D)
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, tasks: List[Task], show_progress: bool = False) -> np.ndarray:
        """
        Encode tasks to embeddings.
        
        Returns:
            (N, 3D) normalized embeddings
        """
        layouts, questions, answers = self._split_components(tasks)
        
        # Each component → D dimensions
        layout_embs = self._encode_texts(layouts, show_progress)      # (N, D)
        question_embs = self._encode_texts(questions, show_progress)  # (N, D)
        answer_embs = self._encode_texts(answers, show_progress)      # (N, D)
        
        # Concatenate: (N, 3D)
        embeddings = np.hstack([layout_embs, question_embs, answer_embs])
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def _encode_texts(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Encode texts to normalized D-dimensional embeddings."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def _split_components(self, tasks: List[Task]) -> tuple:
        """Split tasks into three component lists."""
        layouts = [self._get_layout(task) for task in tasks]
        questions = [task.question for task in tasks]
        answers = [task.answer for task in tasks]
        return layouts, questions, answers
    
    @staticmethod
    def _get_layout(task: Task) -> str:
        """Extract instantiated layout description (placeholders filled)."""
        if not task.metadata:
            raise ValueError(f"Task metadata is missing for task: {task.question[:100]}")
        
        template = task.metadata.get("layout_description", "")
        if not template:
            raise ValueError(f"Layout description is empty for task: {task.question[:100]}")
        
        objects = task.metadata.get("objects", [])
        
        # Replace [obj_0], [obj_1], ... with actual object names
        description = template
        for obj in objects:
            placeholder = f"[{obj['placeholder']}]"
            description = description.replace(placeholder, obj['name'])
        
        return description

