# Alternatives to Shallow Autoencoders for Collaborative Filtering

https://link.springer.com/chapter/10.1007/978-3-032-02215-8_10

Mallea, M., Nebot, À., Mugica, F. (2026). Alternatives to Shallow Autoencoders for Collaborative Filtering. In: Leung, C.K., Dignös, A., Kotsis, G., Tjoa, A.M., Khalil, I. (eds) Big Data Analytics and Knowledge Discovery. DaWaK 2025. Lecture Notes in Computer Science, vol 16048. Springer, Cham. https://doi.org/10.1007/978-3-032-02215-8_10

# Abstract

Collaborative filtering (CF) is a cornerstone of recommender systems and plays a relevant role in many modern applications. CF uses user-item interaction data to discover future preferences. Although deep learning models have shown promise in CF, Embarrassingly Shallow Autoencoders for Sparse Data (EASE) has gained attention for its outstanding ranking accuracy provided by its closed-form solution. EASE relies primarily on relationships among items to fit a full-rank high-dimensional linear mapping. We hypothesize that this design limits its capacity to capture similar but not equivalent fine-grained user relationships, consequently limiting its recommendation accuracy. This paper introduces an alternative formulation based on EASE that takes advantage of user-user information. Furthermore, we propose a hybrid model that combines both user-user and item-item adjacency distributions.

Our experiments reveal that the proposed models outperform EASE on well-known recommendation benchmarks, highlighting the significance of including the user alternative in shallow autoencoder studies for CF.

## EASE_R Model

The **EASE_R** model is a variation of the EASE (Embarrassingly Shallow Autoencoder) algorithm for recommendation systems. It applies a regularization term to the right-side Gram matrix.

### Implementation Details
- Computes the right Gram matrix: \( G_R = X X^T \)
- Applies a regularization parameter \( \lambda \)
- Computes the inverse of \( G_R \)
- Generates item recommendations using:  
  \[
  B_R = P / (-\text{diag}(P))
  \]
  where \( P = G_R^{-1} \).
- Final rating predictions are obtained as:
  \[
  \text{rating} = B_R \cdot X
  \]

### Parameters
- `lambda_`: Regularization parameter.

### Code Reference
Implemented in the `EASE` class under the mode `"ease_R"`.

---

## both_RT Model

The **both_RT** model extends **EASE** by applying regularization to both left and right Gram matrices and performing additional transformations.

### Implementation Details
- Computes both Gram matrices:
  - Left Gram matrix: \( G_L = X^T X \)
  - Right Gram matrix: \( G_R = X X^T \)
- Applies different regularization parameters \( \lambda_L \) and \( \lambda_R \).
- Computes their inverses.
- Generates predictions using matrix multiplications:
  \[
  B_L = G_L^{-1} \cdot X^T
  \]
  \[
  B_R = G_R^{-1} \cdot X
  \]
  \[
  \text{rating} = B_R \cdot X \cdot B_L
  \]

### Parameters
- `lambda_L`: Regularization parameter for left Gram matrix.
- `lambda_R`: Regularization parameter for right Gram matrix.

### Code Reference
Implemented in the `EASE` class under the mode `"both_RT"`.

---

For more details, please refer to the source code in the repository.


See the following repository to get an well-structured implementation:

https://github.com/seoyoungh/svd-ae.git
