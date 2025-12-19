import torch


class MultiModalCollator:
    def collate(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """
        batch: [(x_q, x_d, y), (x_q, x_d, y), ...]
        """
        x_queries, x_documents, ys = zip(*batch)
        
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

        y_batch = torch.stack(ys)

        return x_query_batch, x_document_batch, y_batch
