# LSTM: methods, validation and final test results

## Method and settings

We integrated an LSTM motion encoder with the shared GC data and training interface. Five observed points cover 3.2 seconds. The shared predictor produces six future positions over 4.8 seconds. The encoder uses four motion features per time step and a hidden size of 64. The shared position, neighbour-attention and waypoint/endpoint branches remain part of the full predictor. This version has no scene-image CNN.

We trained eight runs with 203,475 training windows and 43,814 validation windows. We used AdamW, batch size 256 and at most 16 epochs. Checkpoints were selected using validation ADE + 0.3 × turning ADE. Scores were measured in pixels.

## Configuration screens and repeated results

At seed 42, reducing the initial learning rate from 0.001 to 0.0005 increased validation ADE from 20.0434 to 20.3101 pixels. For the two-layer model, adding recurrent dropout 0.1 increased ADE from 20.0331 to 20.0472 pixels. These two screening settings each used one seed.

We repeated the one-layer baseline and the two-layer model without recurrent dropout with seeds 42, 43 and 44. The table reports mean ± sample standard deviation, with ddof = 1.

| Configuration | Validation ADE (px) | Validation FDE (px) | Turning ADE (px) | Parameters |
| --- | --- | --- | --- | --- |
| One-layer baseline | 19.9981 ± 0.0408 | 34.5015 ± 0.0905 | 31.0896 ± 0.2388 | 84,929 |
| Two layers, no recurrent dropout | 20.0423 ± 0.0201 | 34.5699 ± 0.0112 | 31.1261 ± 0.1768 | 118,209 |

The one-layer model had slightly lower mean errors. It performed better at seeds 43 and 44, while the two-layer model performed better at seed 42. Adding a second layer increased total parameters by 39.19% and mean logged training-loop time by 45.51%. The timer includes per-epoch validation and is not an inference-latency measurement.

We therefore retain the one-layer LSTM with hidden size 64 and initial learning rate 0.001 for this experiment budget. Its effective recurrent dropout is zero. The shared fusion branch still uses dropout 0.05.

## Final test results for the frozen configuration

After selecting the one-layer configuration using validation results, we scored the three existing baseline checkpoints on the same complete test split. Each evaluation contains 45,454 windows, including 8,309 turning windows. Every test record matches the corresponding training protocol signature and the shared data fingerprint. We retain all three seeds; no seed or configuration was selected by test performance.

| Training seed | Test ADE (px) | Test FDE (px) | Turning ADE (px) | Turning FDE (px) |
| --- | --- | --- | --- | --- |
| 42 | 21.6442 | 37.4565 | 31.9975 | 61.2750 |
| 43 | 21.5481 | 37.1846 | 31.7055 | 60.2942 |
| 44 | 21.4690 | 37.0783 | 31.6431 | 60.4244 |
| Mean ± sample SD | 21.5538 ± 0.0877 | 37.2398 ± 0.1951 | 31.7821 ± 0.1892 | 60.6646 ± 0.5326 |

The mean test ADE is 1.5557 pixels higher than the mean validation ADE; these are different sample sets. Turning-window errors remain higher than overall errors. The standard deviations describe variation across three trained checkpoints on one test set, not across three independent datasets. We do not use these test scores for further tuning. For the team's seed-42 comparison, use test ADE 21.6442 and FDE 37.4565 pixels alongside matched seed-42 results from the other encoders.

## Limits and attribution

The differences are small. Three seeds do not establish statistical significance or transfer to other scenes. Layer-count changes also affect initialization in the supplied shared factory, so this is a configuration comparison rather than a strict intervention with identical shared initial tensors.

Configuration selection used validation metrics; final test metrics are reported separately. Original training JSON files retain test_evaluated=false because scoring was performed later and saved in separate test.json files. The shared team guide notes earlier development on this data split; the final team report should disclose that history and should not claim that the split was never inspected during all prior development. The LSTM integration and experiments use the supplied shared pipeline, and this model is not a reproduction of Social LSTM.

## References

- Hochreiter and Schmidhuber, Long Short-Term Memory, Neural Computation, 1997. [DOI](https://doi.org/10.1162/neco.1997.9.8.1735).
- Alahi et al., Social LSTM: Human Trajectory Prediction in Crowded Spaces, CVPR 2016. [Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html).
- Yi, Li and Wang, Understanding Pedestrian Behaviors from Stationary Crowd Groups, CVPR 2015. [Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.pdf).
