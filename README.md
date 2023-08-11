# Multi-dimension-index-recommendation
Multi-dimension-index-recommendation
在 common.py 中添加了 block 类的实现，在 test.py 中验证 dmv-tiny-sort.csv 的数据分块后查询是否需要扫描

```bash
python train_SortOrder.py --dataset randomwalk --check_val 5 --epochs 5000 \
--pad_size 1000 --train_block_size 50 --test_block_size 200

```