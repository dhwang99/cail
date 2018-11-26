# cail
司法部智能司法比赛项目,评测指标为 M-f1=(macro-f1+micro-f1)/2。用sklearn跑svm、lr、rf等，最好的还是svm, M-f1为0.81;  的用pytorch跑 textcnn、lstm、rcnn、fasttext等，单模型能到87%左右，模型效果差异不太明显，ensamble后的需要进一步调整。
