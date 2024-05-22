import logging
import torch
import json
import glob
import numpy as np
from tqdm import tqdm
from scipy.sparse import load_npz, csr_array, vstack
from typing import Any, Union
from torch import Tensor as T
from .base import SearchResults, Index

logger = logging.getLogger(__name__)


def scipy_csr_to_torch_csr(mat: np.array, device="cpu"):
    mat_csr = torch.sparse_csr_tensor(
        torch.from_numpy(mat.indptr),
        torch.from_numpy(mat.indices),
        torch.from_numpy(mat.data),
        size=mat.shape,
    )
    mat_csr = mat_csr.to(device)
    return mat_csr

class BinaryTokenIndex(Index):
    """
    Binary token index.
    """
    def __init__(
            self, 
            index_file: str = None,  
            fp16: bool = True, 
            shift: int = 0, 
            device: str = "cpu", 
            data_file: str = None,
            **kwargs):
        self.data = None
        self.index = None
        self.init_index(index_file, fp16, shift, device)
        if data_file:
            self.load_data(data_file)

    def init_index(self, files, fp16, shift, device):
        files = sorted(glob.glob(files))
        logger.info(f"***** Loading index ({len(files)} shards in total) *****")
        shards = [load_npz(f)[:, shift:] for f in files]
        if len(shards) > 1:
            vectors = vstack(shards)
        else:
            vectors = shards[0]
        if fp16:
            vectors = vectors.astype(np.float16)
        logger.info(f"***** Converting index to torch CSR Tensor *****")
        vectors = scipy_csr_to_torch_csr(vectors, device=device)
        vectors = vectors.to(device)
        self.vectors = vectors
        self.device = device

    def to_cpu(self):
        self.vectors = self.vectors.to("cpu")
        self.device = "cpu"

    def to_cuda(self, device="cuda"):
        logger.info(f"***** Moving vector store to device: {device} *****")
        self.vectors = self.vectors.to(device)
        self.device = device

    def load_data(self, file):
        self.data = [json.loads(l) for l in open(file, 'r')]

    def _cpu_search(self, q_embs: T, k: int) -> SearchResults:
        q_embs = csr_array(q_embs.cpu())
        scores = q_embs @ self.vectors.transpose()
        scores_topk = torch.Tensor(scores.toarray()).topk(k)
        search_results = SearchResults(scores_topk.indices.cpu(), scores_topk.values.cpu())
        return search_results

    def _gpu_search(self, q_embs: T, k: int) -> SearchResults:
        q_embs = q_embs.to(self.device).type(self.vectors.dtype)
        with torch.no_grad():
            scores = torch.matmul(self.vectors, q_embs.t()).t()
        scores_topk = scores.topk(k)
        search_results = SearchResults(scores_topk.indices, scores_topk.values)
        return search_results

    def search(self, q_embs: T, k: int) -> SearchResults:
        if self.device == "cpu":
            results = self._cpu_search(q_embs, k)
        else:
            results = self._gpu_search(q_embs, k)
        return results

    def print_info(self):
        logger.info(f"***** Index Info *****")
        logger.info(f"* Type: {self.vectors.dtype}")
        logger.info(f"* Shape: {self.vectors.shape}")
        logger.info(f"* Device: {self.device}")
        logger.info(f"**********************")

    def __len__(self):
        return len(self.data) if self.data else 0
