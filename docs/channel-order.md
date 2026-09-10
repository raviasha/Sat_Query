# Channel-order evidence

Checked 2026-09-09, before implementing the loader.

The CROMA paper, section 3 and appendix A.3.1, identifies SSL4EO-S12 Sentinel-2 L2A and Sentinel-1 GRD as its pretraining inputs. Its public inference code fixes the channel counts to 12 and 2 but does not name the bands individually.

The original SSL4EO-S12 dataset loader explicitly defines and reads this sequence:

| Modality | Channel sequence used here |
| --- | --- |
| Optical | B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 |
| SAR | VV, VH |

BigEarthNet's leading zero is only a filename convention (`B1` becomes `B01`). B8A stays between B08 and B09. Alphabetical file sorting would incorrectly put B8A last and VH before VV.

Sources:

- [CROMA paper, section 3](https://arxiv.org/pdf/2311.00566).
- [Original SSL4EO-S12 loader at pinned revision 2156913](https://github.com/zhu-xlab/SSL4EO-S12/blob/2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py): `ALL_BANDS_S2_L2A`, `ALL_BANDS_S1_GRD`, and `get_array`.
- [CROMA inference interface at pinned revision 59505a6](https://github.com/antofuller/CROMA/blob/59505a6bcadbf36ba20767270154bf9f3067c5e7/use_croma.py).
- [CROMA normalization example at the same revision](https://github.com/antofuller/CROMA/blob/59505a6bcadbf36ba20767270154bf9f3067c5e7/README.md).

Evidence scope: the source dataset order is directly verified; using it for CROMA is an inference from the documented pretraining source. Neither an explicit checkpoint band list nor the authors' complete training loader was found. The manifest records this distinction and does not claim that an inference run or checkpoint-specific validation has occurred.

Implementation decision: use the above pinned, source-backed profile for export rather than an unrecorded guessed sorting order. This refines the proposed plan's binary verified/unverified flag into an evidence-level field. Raw and normalized outputs both preserve the channel sequence and evidence. The normalization policy applies the official README function separately at N=1, with an explicit constant-channel fallback, so batch composition cannot affect a sample.
