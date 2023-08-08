import torch as th
from dgl.nn.pytorch import EdgeWeightNorm
import dgl.function as fn
from torch import nn


class GRecConv(nn.Module):
    def __init__(self, layer, alpha, edge_drop=0):
        super(GRecConv, self).__init__()
        self._layer = layer
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is None:
                src_norm = th.pow(
                    graph.out_degrees().to(feat).clamp(min=1), -0.5
                )
                shp = src_norm.shape + (1,) * (feat.dim() - 1)
                src_norm = th.reshape(src_norm, shp).to(feat.device)
                dst_norm = th.pow(
                    graph.in_degrees().to(feat).clamp(min=1), -0.5
                )
                shp = dst_norm.shape + (1,) * (feat.dim() - 1)
                dst_norm = th.reshape(dst_norm, shp).to(feat.device)
            else:
                edge_weight = EdgeWeightNorm("both")(graph, edge_weight)
            last_feat = feat
            for _ in range(self._layer):
                # normalization by src node
                if edge_weight is None:
                    feat = feat * src_norm
                graph.ndata["h"] = feat
                w = (
                    th.ones(graph.number_of_edges(), 1)
                    if edge_weight is None
                    else edge_weight
                )
                if self.training:
                    graph.edata["w"] = self.edge_drop(w).to(feat.device)
                else:
                    graph.edata["w"] = w.to(feat.device)
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
                feat = graph.ndata.pop("h")
                # normalization by dst node
                if edge_weight is None:
                    feat = feat * dst_norm
                feat = (1 - self._alpha) * feat + self._alpha * last_feat
                last_feat = feat
            return feat
