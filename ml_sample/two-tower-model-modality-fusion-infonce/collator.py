import torch


class MultiModalCollator:
    def collate(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        batch: [(x_q, x_d), (x_q, x_d), ...]
        """
        x_queries, x_documents, x_hard_negative_documents = zip(*batch)
        
        # --- Query 側の処理 ---
        q_keys = x_queries[0].keys()
        x_query_batch = {}
        for key in q_keys:
            vals = [q[key] for q in x_queries]
            x_query_batch[key] = torch.stack(vals)

        # --- Document 側の処理 ---
        d_keys = x_documents[0].keys()
        x_document_batch = {}
        for key in d_keys:
            vals = [d[key] for d in x_documents]
            x_document_batch[key] = torch.stack(vals)
        
        d_keys = x_hard_negative_documents[0].keys()
        x_hard_negative_document_batch = {}
        for key in d_keys:
            vals = [d[key] for d in x_hard_negative_documents]
            x_hard_negative_document_batch[key] = torch.stack(vals)

        return x_query_batch, x_document_batch, x_hard_negative_document_batch
