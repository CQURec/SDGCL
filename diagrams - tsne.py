import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#添加对抗辅助模块
X = np.load('no_adv_feature_0.npy')  
y = np.load('no_adv_label_0.npy')  


tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_embedded = tsne.fit_transform(X)


rgb_color1 = [(236/255, 100/255, 75/255)]  # 将 RGB 颜色值转换为 0 到 1 的比例
rgb_color2 = [(43/255, 44/255, 170/255)]

plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='blue',s=15, label='non-rumor')
plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='red',s=15, label='rumor')
# plt.title('t-SNE Visualization with Labels')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()



#不添加对抗辅助模块
X = np.load('feature_1.npy') 
y = np.load('label_1.npy')  


tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_embedded = tsne.fit_transform(X)


plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='blue',s=15, label='non-rumor')
plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='red',s=15, label='rumor')
# plt.title('t-SNE Visualization with Labels')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()