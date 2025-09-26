1. 百分比使用数据集 done
3. 自动按数据集进行对比 done
4. 结合config，evaluation生成图像 done
5. 增加checkpoint功能，每次有超过val准确率的都要保存一次 done
6. 为每个n个epoch的验证也标准化为类似model加载的模块化功能，函数可以和eval的通用 done
7. 只要进行了每个n个epoch的验证或者训练结束后的验证都要保存，并且重开训练不会覆盖上一次训练的记录（使用yaml文件名+v0/v1/v2做到）done

TO DO:
1. 自动参数寻优
2. csv文件格式数据集兼容
3. 兼容同数据集不同evaluation方式（不确定）
4. 扩展模型
5. 扩展数据集
6. 兼容sequence2sequence格式模型
7. 兼容单/多输出回归，时序预测功能（不确定）