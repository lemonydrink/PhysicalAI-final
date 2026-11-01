# Third-party components

The root LICENSE applies to this repository's original contributions. It does not replace upstream licenses or model/dataset terms.

- `workspace/backends/smolvla/third_party/lerobot`: Hugging Face LeRobot snapshot with local compatibility patches. Its Apache-2.0 LICENSE is retained. Revision, hashes and patches are recorded in `source.json` and `compat/`.
- `workspace/third_party/libero`: local LIBERO-plus source snapshot. Original attribution and headers are retained; see `workspace/docs/LIBERO_SOURCE.json` and the upstream README for origin. No new license grant for this third-party code is implied by the root LICENSE.
- `workspace/src/physicalai_vla/upstream`: X-VLA reference implementation. Provenance is in `workspace/docs/UPSTREAM_SOURCE.json`; retained notices and upstream terms apply.
- BurgQ model, SmolVLM model and LIBERO datasets are referenced by repo/revision. Model weights and dataset frames are not redistributed here. Obtain these under their upstream terms.
- Included simulator rollout videos are outputs of the reported local experiment. Simulator scene content retains its upstream attribution.
