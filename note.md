1. 百分比使用数据集 done
3. 自动按数据集进行对比 done
4. 结合config，evaluation生成图像 done
5. 增加checkpoint功能，每次有超过val准确率的都要保存一次 done
6. 为每个n个epoch的验证也标准化为类似model加载的模块化功能，函数可以和eval的通用 done
7. 只要进行了每个n个epoch的验证或者训练结束后的验证都要保存，并且重开训练不会覆盖上一次训练的记录（使用yaml文件名+v0/v1/v2做到）done

TO DO:
1. 自动参数寻优 done
2. csv文件格式数据集兼容 to do
3. 设置全局seed to do
4. 扩展模型 todo
5. 扩展数据集 todo
6. 兼容sequence2sequence格式模型 done
7. 直接集成论文中的模型 patchTST
8. excel输出格式 done


3. 兼容同数据集不同evaluation方式（不确定）
7. 兼容单/多输出回归，时序预测功能（不确定）

进度：
===========================
1. 批量导入和训练数据集
2. 集成patchTST
3. grid search完成
4. 结果格式改善（新增错误日志， excel表格分类）
5. 新增trainer进行不同训练模式

目前模型： LSTM/CNN/PatchTST 缺少sota模型
数据集: UCR2018(128), UCI_HAR, NASA_IMS 缺少故障诊断领域数据集

1. comforamal learning + welding 
2. pipeline to english
3. LLM 记录 done
4. 周四, am 11.30 会议


