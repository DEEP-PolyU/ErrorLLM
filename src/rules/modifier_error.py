from typing import Dict, List
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector


class ModifierErrorDetector(BaseDetector):
    
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []
        
        return errors
