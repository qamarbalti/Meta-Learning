On the Algorithmic Implementation of
Multiclass Kernel-based Vector Machines
Koby Crammer kobics@cs.huji.ac.il
Yoram Singer singer@cs.huji.ac.il
School of Computer Science & Engineering
Hebrew University, Jerusalem 91904, Israel
Editors: Nello Cristianini, John Shawe-Taylor and Bob Williamson
Abstract
In this paper we describe the algorithmic implementation of multiclass kernel-based vector
machines. Our starting point is a generalized notion of the margin to multiclass problems.
Using this notion we cast multiclass categorization problems as a constrained optimization
problem with a quadratic objective function. Unlike most of previous approaches which
typically decompose a multiclass problem into multiple independent binary classification
tasks, our notion of margin yields a direct method for training multiclass predictors. By
using the dual of the optimization problem we are able to incorporate kernels with a
compact set of constraints and decompose the dual problem into multiple optimization
problems of reduced size. We describe an efficient fixed-point algorithm for solving the
reduced optimization problems and prove its convergence. We then discuss technical details
that yield significant running time improvements for large datasets. Finally, we describe
various experiments with our approach comparing it to previously studied kernel-based
methods. Our experiments indicate that for multiclass problems we attain state-of-the-art
accuracy.
Keywords: Multiclass problems, SVM, Kernel Machines
1. Introduction
Supervised machine learning tasks often boil down to the problem of assigning labels to
instances where the labels are drawn from a finite set of elements. This task is referred
to as multiclass learning. Numerous specialized algorithms have been devised for multiclass problems by building upon classification learning algorithms for binary problems, i.e.,
problems in which the set of possible labels is of size two. Notable examples for multiclass
learning algorithms are the multiclass extensions for decision tree learning (Breiman et al.,
1984, Quinlan, 1993) and various specialized versions of boosting such as AdaBoost.M2 and
AdaBoost.MH (Freund and Schapire, 1997, Schapire and Singer, 1999). However, the dominating approach for solving multiclass problems using support vector machines has been
based on reducing a single multiclass problems into multiple binary problems. For instance,
a common method is to build a set of binary classifiers where each classifier distinguishes
between one of the labels to the rest. This approach is a special case of using output codes
for solving multiclass problems (Dietterich and Bakiri, 1995). However, while multiclass
learning using output codes provides a simple and powerful framework it cannot capture
c 2001 Koby Crammer and Yoram Singer.
Crammer and Singer
correlations between the different classes since it breaks a multiclass problem into multiple
independent binary problems.
In this paper we develop and discuss in detail a direct approach for learning multiclass
support vector machines (SVM). SVMs have gained an enormous popularity in statistics,
learning theory, and engineering (see for instance Vapnik, 1998, Sch¨olkopf et al., 1998, Cristianini and Shawe-Taylor, 2000, and the many references therein). With a few exceptions
most support vector learning algorithms have been designed for binary (two class) problems. A few attempts have been made to generalize SVM to multiclass problems (Weston
and Watkins, 1999, Vapnik, 1998). These attempts to extend the binary case are acheived
by adding constraints for every class and thus the size of the quadratic optimization is
proportional to the number categories in the classification problems. The result is often a
homogeneous quadratic problem which is hard to solve and difficult to store.
The starting point of our approach is a simple generalization of separating hyperplanes
and, analogously, a generalized notion of margins for multiclass problems. This notion of a
margin has been employed in previous research (Allwein et al., 2000) but not in the context
of SVM. Using the definition of a margin for multiclass problems we describe in Section 3 a
compact quadratic optimization problem. We then discuss its dual problem and the form of
the resulting multiclass predictor. In Section 4 we give a decomposition of the dual problem
into multiple small optimization problems. This decomposition yields a memory and time
efficient representation of multiclass problems. We proceed and describe an iterative solution
for the set of the reduced optimization problems. We first discuss in Section 5 the means of
choosing which reduced problem to solve on each round of the algorithm. We then discuss
in Section 6 an efficient fixed-point algorithm for finding an approximate solution for the
reduced problem that was chosen. We analyze the algorithm and derive a bound on its rate
of convergence to the optimal solution. The baseline algorithm is based on a main loop which
is composed of an example selection for optimization followed by an invocation of the fixedpoint algorithm with the example that was chosen. This baseline algorithm can be used
with small datasets but to make it practical for large ones, several technical improvements
had to be sought. We therefore devote Section 7 to a description of the different technical
improvements we have taken in order to make our approach applicable to large datasets. We
also discuss the running time and accuracy results achieved in experiments that underscore
the technical improvements. In addition, we report in Section 8 the results achieved in
evaluation experiments, comparing them to previous work. Finally, we give conclusions in
Section 9.
Related work Naturally, our work builds on previous research and advances in learning
using support vector machines. The space is clearly too limited to mention all the relevant
work, and thus we refer the reader to the books and collections mentioned above. As we
have already mentioned, the idea of casting multiclass problems as a single constrained optimization with a quadratic objective function was proposed by Vapnik (1998), Weston and
Watkins (1999), Bredensteiner and Bennet (1999), and Guermeur et. al (2000). However,
the size of the resulting optimization problems devised in the above papers is typically large
and complex. The idea of breaking a large constrained optimization problem into small
problems, where each of which employs a subset of the constraints was first explored in
the context of support vector machines by Boser et al. (1992). These ideas were further
266
Multiclass Kernel-based Vector Machines
developed by several researchers (see Joachims, 1998 for an overview). However, the roots
of this line of research go back to the seminal work of LevBregman (1967) which was further developed by Yair Censor and colleagues (see Censor and Zenios, 1997 for an excellent
overview). These ideas distilled in Platt’s method, called SMO, for sequential minimal optimization. SMO works with reduced problems that are derived from a pair of examples
while our approach employs a single example for each reduced optimization problem. The
result is a simple optimization problem which can be solved analytically in binary classification problems (see Platt, 1998) and leads to an efficient numerical algorithm (that is
guaranteed to converge) in multiclass settings. Furthermore, although not explored in this
paper, it deems possible that the single-example reduction can be used in parallel applications. Many of the technical improvements we discuss in this paper have been proposed
in previous work. In particular ideas such as using a working set and caching have been
described by Burges (1998), Platt (1998), Joachims (1998), and others. Finally, we would
like to note that this work is part of a general line of research on multiclass learning we
have been involved with. Allwein et al. (2000) described and analyzed a general approach
for multiclass problems using error correcting output codes (Dietterich and Bakiri, 1995).
Building on that work, we investigated the problem of designing good output codes for multiclass problems (Crammer and Singer, 2000). Although the model of learning using output
codes differs from the framework studied in this paper, some of the techniques presented in
this paper build upon results from an earlier paper (Crammer and Singer, 2000). Finally,
some of the ideas presented in this paper can also be used to build multiclass predictors
in online settings using the mistake bound model as the means of analysis. Our current
research on multiclass problems concentrates on analogous online approaches (Crammer
and Singer, 2001).
2. Preliminaries
Let S = {(¯x1, y1),...,(¯xm, ym)} be a set of m training examples. We assume that each
example ¯xi is drawn from a domain X ⊆n and that each label yi is an integer from the
set Y = {1,...,k}. A (multiclass) classifier is a function H : X→Y that maps an instance
x¯ to an element y of Y. In this paper we focus on a framework that uses classifiers of the
form
HM(¯x) = arg k
max
r=1 {M¯r · x¯} ,
where M is a matrix of size k × n over  and M¯r is the rth row of M. We interchangeably
call the value of the inner-product of the rth row of M with the instance ¯x the confidence
and the similarity score for the r class. Therefore, according to our definition above, the
predicted label is the index of the row attaining the highest similarity score with ¯x. This
setting is a generalization of linear binary classifiers. Using the notation introduced above,
linear binary classifiers predict that the label of an instance ¯x is 1 if ¯w·x >¯ 0 and 2 otherwise
( ¯w ·x¯ ≤ 0). Such a classifier can be implemented using a matrix of size 2×n where M¯ 1 = ¯w
and M¯ 2 = −w¯. Note, however, that this representation is less efficient as it occupies twice
the memory needed. Our model becomes parsimonious when k ≥ 3 in which we maintain k
prototypes M¯ 1, M¯ 2,..., M¯k and set the label of a new input instance by choosing the index
of the most similar row of M.
267
Crammer and Singer
Figure 1: Illustration of the margin bound employed by the optimization problem.
Given a classifier HM(¯x) (parametrized by a matrix M) and an example (¯x, y), we say
that HM(¯x) misclassifies an example ¯x if HM(¯x) 
= y. Let [[π]] be 1 if the predicate π holds
and 0 otherwise. Thus, the empirical error for a multiclass problem is given by
S(M) = 1
m
m
i=1
[[HM(xi) 
= yi]] . (1)
Our goal is to find a matrix M that attains a small empirical error on the sample S and
also generalizes well. Direct approaches that attempt to minimize the empirical error are
computationally expensive (see for instance H¨offgen and Simon, 1992, Crammer and Singer,
2000). Building on Vapnik’s work on support vector machines (Vapnik, 1998), we describe
in the next section our paradigm for finding a good matrix M by replacing the discrete
empirical error minimization problem with a quadratic optimization problem. As we see
later, recasting the problem as a minimization problem also enables us to replace innerproducts of the form ¯a·¯b with kernel-based inner-products of the form K(¯a, ¯b) = φ¯(¯a)·φ¯(¯b).
3. Constructing multiclass kernel-based predictors
To construct multiclass predictors we replace the misclassification error of an example,
([[HM(x) 
= y]]), with the following piecewise linear bound,
maxr {M¯r · x¯ + 1 − δy,r} − M¯ y · x , ¯
where δp,q is equal 1 if p = q and 0 otherwise. The above bound is zero if the confidence
value for the correct label is larger by at least one than the confidences assigned to the rest
of the labels. Otherwise, we suffer a loss which is linearly proportional to the difference
between the confidence of the correct label and the maximum among the confidences of the
other labels. A graphical illustration of the above is given in Figure 1. The circles in the
figure denote different labels and the correct label is plotted in dark grey while the rest
of the labels are plotted in light gray. The height of each label designates its confidence.
Three settings are plotted in the figure. The left plot corresponds to the case when the
margin is larger than one, and therefore the bound maxr{M¯r · x¯ + 1 − δy,r} − M¯ y · x¯ equals
zero, and hence the example is correctly classified. The middle figure shows a case where
the example is correctly classified but with a small margin and we suffer some loss. The
right plot depicts the loss of a misclassified example.
268
Multiclass Kernel-based Vector Machines
Summing over all the examples in S we get an upper bound on the empirical loss,
S(M) ≤
1
m
m
i=1

maxr {M¯r · x¯i + 1 − δyi,r} − M¯ yi · x¯i

. (2)
We say that a sample S is linearly separable by a multiclass machine if there exists a
matrix M such that the above loss is equal to zero for all the examples in S, that is,
∀i maxr {M¯r · x¯i + 1 − δyi,r} − M¯ yi · x¯i = 0 . (3)
Therefore, a matrix M that satisfies Eq. (3) would also satisfy the constraints,
∀i, r M¯ yi · x¯i + δyi,r − M¯r · x¯i ≥ 1 . (4)
Define the l2-norm of a matrix M to be the l2-norm of the vector represented by the concatenation of M’s rows, M2
2 = (M¯1,..., M¯k)2
2 = 
i,j M2
i,j . Note that if the constraints
given by Eq. (4) are satisfied, we can make the differences between M¯ yi ·x¯i and M¯r ·x¯i arbitrarily large. Furthermore, previous work on the generalization properties of large margin
DAGs (Platt et al., 2000) for multiclass problems showed that the generalization properties
depend on the l2-norm of M (see also Crammer and Singer, 2000). We therefore would like
to seek a matrix M of a small norm that satisfies Eq. (4). When the sample S is linearly
separable by a multiclass machine, we seek a matrix M of the smallest norm that satisfies
Eq. (4). The result is the following optimization problem,
min
M
1
2
M2
2 (5)
subject to : ∀i, r M¯ yi · x¯i + δyi,r − M¯r · x¯i ≥ 1 .
Note that m of the constraints for r = yi are automatically satisfied since,
M¯ yi · x¯i + δyi,yi − M¯ yi · x¯i = 1 .
This property is an artifact of the separable case. In the general case the sample S might
not be linearly separable by a multiclass machine. We therefore add slack variables ξi ≥ 0
and modify Eq. (3) to be,
∀i maxr {M¯r · x¯i + 1 − δyi,r} − M¯ yi · x¯i = ξi . (6)
We now replace the optimization problem defined by Eq. (5) with the following primal
optimization problem,
min
M,ξ
1
2
βM2
2 +m
i=1
ξi (7)
subject to : ∀i, r M¯ yi · x¯i + δyi,r − M¯r · x¯i ≥ 1 − ξi .
where β > 0 is a regularization constant and for r = yi the inequality constraints become
ξi ≥ 0. This is an optimization problem with “soft” constraints. We would like to note in
269
Crammer and Singer
passing that it is possible to cast an analogous optimization problem with “hard” constraints
as in (Vapnik, 1998).
To solve the optimization problem we use the Karush-Kuhn-Tucker theorem (see for
instance Vapnik, 1998, Cristianini and Shawe-Taylor, 2000). We add a dual set of variables,
one for each constraint and get the Lagrangian of the optimization problem,
L(M, ξ,η) = 1
2
β

r
M¯r2
2 +m
i=1
ξi (8)
+

i,r
ηi,r 
M¯r · x¯i − M¯ yi · x¯i − δyi,r + 1 − ξi

subject to : ∀i, r ηi,r ≥ 0 .
We now seek a saddle point of the Lagrangian, which would be the minimum for the primal
variables {M, ξ} and the maximum for the dual variables η. To find the minimum over the
primal variables we require,
∂
∂ξi
L = 1 −
r
ηi,r = 0 ⇒ 
r
ηi,r = 1 . (9)
Similarly, for M¯r we require,
∂
∂M¯r
L = 
i
ηi,rx¯i − 
i,yi=r

q
ηi,q
	 
  =1
x¯i + βM¯r
= 
i
ηi,rx¯i −
i
δyirx¯i + βM¯r = 0 ,
which results in the following form
M¯r = β−1



i
(δyi,r − ηi,r)¯xi

. (10)
Eq. (10) implies that the solution of the optimization problem given by Eq. (5) is a matrix
M whose rows are linear combinations of the instances ¯x1 ... x¯m. Note that from Eq. (10)
we get that the contribution of an instance ¯xi to M¯r is δyi,r − ηi,r. We say that an example
x¯i is a support pattern if there is a row r for which this coefficient is not zero. For each
row M¯r of the matrix M we can partition the patterns with nonzero coefficients into two
subsets by rewriting Eq. (10) as follows,
M¯r = β−1

 
i:yi=r
(1 − ηi,r)¯xi + 
i:yi=r
(−ηi,r)¯xi

 .
The first sum is over all patterns that belong to the rth class. Hence, an example ¯xi labeled
yi = r is a support pattern only if ηi,r = ηi,yi < 1. The second sum is over the rest of the
patterns whose labels are different from r. In this case, an example ¯xi is a support pattern
270
Multiclass Kernel-based Vector Machines
only if ηi,r > 0. Put another way, since for each pattern ¯xi the set {ηi,1, ηi,2,...,ηi,k} satisfies
the constraints ηi,1,...,ηi,k ≥ 0 and 
r ηi,r = 1, each set can be viewed as a probability
distribution over the labels {1 ...k}. Under this probabilistic interpretation an example
x¯i is a support pattern if and only if its corresponding distribution is not concentrated on
the correct label yi. Therefore, the classifier is constructed using patterns whose labels are
uncertain; the rest of the input patterns are ignored.
Next, we develop the Lagrangian using only the dual variables by substituting Eqs. (9)
and (10) into Eq. (8). Since the derivation is rather technical we defer the complete derivation to App. A. We obtain the following objective function of the dual program,
Q(η) = −1
2
β−1
i,j
(¯xi · x¯j )



r
(δyi,r − ηi,r)(δyj ,r − ηj,r)

−
i,r
ηi,rδyi,r .
Let ¯1i be the vector whose components are all zero except for the ith component which is
equal to one, and let ¯1 be the vector whose components are all one. Using this notation we
can rewrite the dual program in the following vector form,
max
η
Q(η) = −1
2
β−1
i,j
(¯xi · x¯j )

(¯1yi − η¯i) · (¯1yj − η¯j )

−
i
η¯i · ¯1yi (11)
subject to : ∀i : ¯ηi ≥ 0 and ¯ηi · ¯1=1 .
It is easy to verify that Q(η) is concave in η. Since the set of constraints is convex, there
is a unique maximum value of Q(η). To simplify the problem we now perform the following
change of variables. Let ¯τi = ¯1yi − η¯i be the difference between the point distribution ¯1yi
concentrating on the correct label and the distribution ¯ηi obtained by the optimization
problem. Then Eq. (10) that describes the form of M becomes,
M¯r = β−1
i
τi,rx¯i . (12)
Since we search for the value of the variables which maximize the objective function Q (and
not the optimum value of Q itself), we can omit any additive and positive multiplicative
constants and write the dual problem given by Eq. (11) as,
maxτ Q(τ ) = −1
2

i,j
(¯xi · x¯j )(¯τi · τ¯j ) + β

i
τ¯i · ¯1yi (13)
subject to : ∀i τ¯i ≤ ¯1yi and ¯τi · ¯1=0 .
Finally, we rewrite the classifier H(¯x) in terms of the variable τ ,
H(¯x) = arg k
max
r=1

M¯r · x¯
 = arg
k
max
r=1 
i
τi,r(¯xi · x¯)

. (14)
As in Support Vector Machines (Cortes and Vapnik, 1995), the dual program and the
resulting classifier depend only on inner products of the form (¯xi · x¯). Therefore, we can
perform inner-product calculations in some high dimensional inner-product space Z by
271
Crammer and Singer
replacing the inner-products in Eq. (13) and in Eq. (14) with a kernel function K(·, ·)
that satisfies Mercer’s conditions (Vapnik, 1998). The general dual program using kernel
functions is therefore,
maxτ Q(τ ) = −1
2

i,j
K (¯xi, x¯j ) (¯τi · τ¯j ) + β

i
τ¯i · ¯1yi (15)
subject to : ∀i τ¯i ≤ ¯1yi and ¯τi · ¯1=0 ,
and the classification rule H(¯x) becomes,
H(¯x) = arg k
max
r=1 
i
τi,rK (¯x, x¯i)

. (16)
Therefore, constructing a multiclass predictor using kernel-based inner-products is as simple
as using standard inner-products.
Note the classifier of Eq. (16) does not contain a bias parameter br for r = 1 ...k.
Augmenting these terms will add m more equality constraints to the dual optimization
problem, increasing the complexity of the optimization problem. However, one can always
use inner-products of the form K(¯a, ¯b) + 1 which is equivalent of using bias parameters, and
adding 1
2β 
r b2
r to the objective function.
Also note that in the special case of k = 2 Eq. (7) reduces to the primal program of SVM
by setting ¯w = M¯ 1 − M¯ 2 and C = β−1. As mentioned above, Weston and Watkins (1999)
also developed a multiclass version for SVM. Their approach compared the confidence M¯ y ·x¯
of the correct label to the confidences of all the other labels M¯r ·x¯ and therefore used m(k−1)
slack variables in the primal problem. In contrast, in our framework the confidence of the
correct label is compared to the highest similarity-score among the rest of the labels and uses
only m slack variables in the primal program. As we describe in the sequel our compact
formalization leads to a memory and time efficient algorithm for the above optimization
problem.
4. Decomposing the optimization problem
The dual quadratic program given by Eq. (15) can be solved using standard quadratic
programming (QP) techniques. However, since it employs mk variables, converting the
dual program given by Eq. (15) into a standard QP form yields a representation that
employs a matrix of size mk × mk, which leads to a very large scale problem in general.
Clearly, storing a matrix of that size is intractable for large problems. We now introduce
a simple, memory efficient algorithm for solving the quadratic optimization problem given
by Eq. (15) by decomposing it into small problems.
The core idea of our algorithm is based on separating the constraints of Eq. (15) into
m disjoint sets, {τ¯i|τ¯i ≤ ¯1yi , τ¯i · ¯1=0}m
i=1 . The algorithm we propose works in rounds.
On each round the algorithm chooses a pattern p and improves the value of the objective
function by updating the variables ¯τp under the set of constraints, ¯τp ≤ ¯1yp and ¯τp · ¯1 = 0.
Let us fix an example index p and write the objective function only in terms of the
variables ¯τp. For brevity we use Ki,j to denote K (¯xi, x¯j ). We now isolate the contribution
272
Multiclass Kernel-based Vector Machines
Input {(¯x1, y1),...,(¯xm, ym)}.
Initialize τ¯1 = ¯0,..., τ¯m = ¯0.
Loop:
1. Choose an example p.
2. Calculate the constants for the reduced problem:
• Ap = K(¯xp, x¯p)
• B¯p = 
i=p K(¯xi, x¯p)¯τi − β¯1yp
3. Set ¯τp to be the solution of the reduced problem :
min
τp
Q(¯τp) = 1
2
Ap(¯τp · τ¯p) + B¯p · τ¯p
subject to : ¯τp ≤ ¯1yp and ¯τp · ¯1=0
Output : H(¯x) = arg k
max
r=1 
i
τi,rK (¯x, x¯i)

.
Figure 2: Skeleton of the algorithm for learning multiclass support vector machine.
of ¯τp in Q.
Qp(¯τp) def
= −1
2

i,j
Ki,j (¯τi · τ¯j ) + β

i
τ¯i · ¯1yi
= −1
2
Kp,p(¯τp · τ¯p) −
i=p
Ki,p(¯τp · τ¯i)
−1
2

i=p,j=p
Ki,j (¯τi · τ¯j ) + βτ¯p · ¯1yp + β

i=p
τ¯i · ¯1yi
= −1
2
Kp,p(¯τp · τ¯p) − τ¯p ·

−β¯1yp +
i=p
Ki,pτ¯i


+

−1
2

i=p,j=p
Ki,j (¯τi · τ¯j ) + β

i=p
τ¯i · ¯1yi

 . (17)
Let us now define the following variables,
Ap = Kp,p > 0 (18)
B¯p = −β¯1yp +
i=p
Ki,pτ¯i (19)
Cp = −1
2

i,j=p
Ki,j (¯τi · τ¯j ) + β

i=p
τ¯i · ¯1yi .
Using the variables defined above the objective function becomes,
Qp(¯τp) = −1
2
Ap(¯τp · τ¯p) − B¯p · τ¯p + Cp .
273
Crammer and Singer
For brevity, let us now omit all constants that do not affect the solution. Each reduced
optimization problem has k variables and k + 1 constraints,
minτ Q(¯τ ) = 1
2
Ap(¯τp · τ¯p) + B¯p · τ¯p (20)
subject to : ¯τp ≤ ¯1yp and ¯τp · ¯1=0 .
The skeleton of the algorithm is given in Figure 2. The algorithm is initialized with
τ¯i = ¯0 for i = 1 ...m which, as we discuss later, leads to a simple initialization of internal
variables the algorithm employs for efficient implementation. To complete the details of the
algorithm we need to discuss the following issues. First, we need a stopping criterion for
the loop. A simple method is to run the algorithm for a fixed number of rounds. A better
approach which we discuss in the sequel is to continue iterating as long as the algorithm
does not meet a predefined accuracy condition. Second, we need a scheme for choosing
the pattern p on each round which then induces the reduced optimization problem given in
Eq. (20). Two commonly used methods are to scan the patterns sequentially or to choose a
pattern uniformly at random. In this paper we describe a scheme for choosing an example
p in a greedy manner. This scheme appears to perform better empirically than other naive
schemes. We address these two issues in Section 5.
The third issue we need to address is how to solve efficiently the reduced problem given
by Eq. (20). Since this problem constitutes the core and the inner-loop of the algorithm we
develop an efficient method for solving the reduced quadratic optimization problem. This
method is more efficient than using the standard QP techniques, especially when it suffices
to find an approximation to the optimal solution. Our specialized solution enables us to
solve problems with a large number of classes k when a straightforward approach could not
be applicable. This method is described in Section 6.
5. Example selection for optimization
To remind the reader, we need to solve Eq. (15),
minτ Q(τ ) = 1
2

i,j
Ki,j (¯τi · τ¯j ) − β

i
τ¯i · ¯1yi
subject to : ∀i τ¯i ≤ ¯1yi and ¯τi · ¯1=0 ,
where as before Ki,j = K (¯xi, x¯j ). We use the Karush-Kuhn-Tucker theorem (see Cristianini
and Shawe-Taylor, 2000) to find the necessary conditions for a point τ to be an optimum
of Eq. (15). The Lagrangian of the problem is,
L(τ, u, v) = 1
2

i,j
Ki,j 
r
τi,rτj,r − β

i,r
τi,rδyi,r (21)
+

i,r
ui,r(τi,r − δyi,r) −
i
vi

r
τi,r
subject to : ∀i, r ui,r ≥ 0 .
The first condition is,
∂
∂τi,r
L = 
j
Ki,jτj,r − βδyi,r + ui,r − vi = 0 . (22)
274
Multiclass Kernel-based Vector Machines
Let us now define the following auxiliary set of variables,
Fi,r = 
j
Ki,jτj,r − βδyi,r . (23)
For each instance ¯xi, the value of Fi,r designates the confidence in assigning the label r to
x¯i. A value of β is subtracted from the correct label confidence in order to obtain a margin
of at least β. Note that from Eq. (19) we get,
Fp,r = Bp,r + kp,pτp,r . (24)
We will make use of this relation between the variables F and B in the next section in
which we discuss an efficient solution to the quadratic problem.
Taking the derivative with respect to the dual variables of the Lagrangian given by
Eq. (21) and using the definition of Fi,r from Eq. (23) and KKT conditions we get the
following set of equality constraints on a feasible solution for the quadratic optimization
problem,
∀i, r Fi,r + ui,r = vi , (25)
∀i, r ui,r(τi,r − δyi,r)=0 , (26)
∀i, r ui,r ≥ 0 . (27)
We now further simplify the equations above. We do so by considering two cases. The first
case is when τi,r = δyi,r. In this case Eq. (26) holds automatically. By combining Eq. (27)
and Eq. (25) we get that,
Fi,r ≤ vi . (28)
In the second case τi,r < δyi,r. In order for Eq. (26) to hold we must have ui,r = 0. Thus,
using Eq. (25) we get that,
Fi,r = vi .
We now replace the single equality constraint with the following two inequalities,
Fi,r ≥ vi and Fi,r ≤ vi . (29)
To remind the reader, the constraints on ¯τ from the optimization problem given by Eq. (15)
imply that for all i, ¯τi ≤ ¯1yi and ¯τi · ¯1 = 0. Therefore, if these constraints are satisfied there
must exist at least one label r for which τi,r < δyi,r. We thus get that vi = maxr Fi,r. Note
also that if ¯τi = 0 then Fi,yi = vi = maxr Fi,r and Fi,yi is the unique maximum. We now
combine the set of constraints from Eqs. (28) and (29) into a single inequality,
maxr Fi,r ≤ vi ≤ min r : τi,r<δyi,r
Fi,r . (30)
Finally, dropping vi we obtain,
maxr Fi,r ≤ min r : τi,r<δyi,r
Fi,r . (31)
We now define,
ψi = maxr Fi,r − min r : τi,r<δyi,r
Fi,r . (32)
275
Crammer and Singer
Since maxr Fi,r ≥ minr : τi,r<δyi,r Fi,r then the necessary and sufficient condition for a feasible vector ¯τi to be an optimum for Eq. (15) is that, ψi = 0. In the actual numerical
implementation it is sufficient to find ¯τi such that ψi ≤  where  is a predefined accuracy
parameter. We therefore keep performing the main loop of Figure 2 so long as there are
examples (¯xi, yi) whose values ψi are greater than .
The variables ψi also serve as our means for choosing an example for an update. In
our implementation we try to keep the memory requirements as small as possible and thus
manipulate a single example on each loop. We choose the example index p for which ψp is
maximal. We then find the vector ¯τp which is the (approximate) solution of the reduced
optimization problem given by Eq. (15). Due to the change in ¯τp we need to update Fi,r
and ψi for all i and r. The pseudo-code describing this process is deferred to the next
section in which we describe a simple and efficient algorithm for finding an approximate
solution for the optimization problem of Eq. (15). Lin (2001) showed that this scheme
does converge to the solution in a finite number of steps. Finally, we would like to note
that some of the underlying ideas described in this section have been also explored by
Keerthi and Gilbert (2000).
6. Solving the reduced optimization problem
The core of our algorithm relies on an efficient method for solving the reduced optimization
given by Eq. (15) or the equivalent problem as defined by Eq. (20). In this section we
describe an efficient fixed-point algorithm that finds an approximate solution to Eq. (20).
We would like to note that an exact solution can also be derived. In (Crammer and Singer,
2000) we described a closely related algorithm for solving a similar quadratic optimization
problem in the context of output coding. A simple modification of the algorithm can be
used here. However, the algorithm needs to sort k values on each iteration and thus might
be slow when k is large. Furthermore, as we discuss in the next section, we found empirically
that the quality of the solution is quite insensitive to how well we fulfill the Karush-KuhnTucker condition by bounding ψi. Therefore, it is enough to find a vector ¯τp that decreases
significantly the value of Q(¯τ ) but is not necessarily the optimal solution.
We start by rewriting Q(¯τ ) from Eq. (20) using a completion to quadratic form and
dropping the pattern index p,
Q(¯τ ) = −1
2
A(¯τ · τ¯) − B¯ · τ¯
= −1
2
A[(¯τ +
B¯
A) · (¯τ +
B¯
A)] + B¯ · B¯
2A .
We now perform the following change of variables,
ν¯ = ¯τ +
B¯
A D¯ = B¯
A + ¯1y . (33)
At this point, we omit additive constants and the multiplicative factor A since they do not
affect the value of the optimal solution. Using the above variable the optimization problem
from Eq. (20) now becomes,
minν Q(¯ν) = ν¯2 (34)
subject to : ¯ν ≤ D¯ and ¯ν · ¯1 = D¯ · ¯1 − 1 .
276
Multiclass Kernel-based Vector Machines
We would like to note that since
Fi,r = Bi,r + Aiτi,r ,
we can compute ψi from B¯i and thus need to store either Bi,r or Fi,r.
Let us denote by θ and αi the variables of the dual problem of Eq. (34). Then, the
Karush-Kuhn-Tucker conditions imply that,
∀r νr ≤ Dr ; αr(νr − Dr)=0 ; νr + αr − θ = 0 . (35)
Note that since αr ≥ 0 the above conditions imply that νr ≤ θ for all r. Combining this
inequality with the constraint that νr ≤ Dr we get that the solution satisfies
νr ≤ min{θ, Dr} . (36)
If αr = 0 we get that νr = θ and if αr > 0 we must have that νr = Dr. Thus, Eq. (36)
holds with equality, namely, the solution is of the form, νr = min{θ, Dr}. Now, since
ν¯ · ¯1 = D¯ · ¯1 − 1 we get that θ satisfies the following constraint,

k
r=1
minr {θ, Dr} = 
k
r=1
Dr − 1 . (37)
The above equation uniquely defines θ since the sum k
r=1 minr{θ, Dr} is a strictly monotone and continuous function in θ. For θ = maxr Dr we have that k
r=1 minr{θ, Dr} >
k
r=1 Dr − 1 while k
r=1 minr{θ, Dr} → −∞ as θ → −∞. Therefore, there always exists a
unique value θ that satisfies Eq. (37). The following theorem shows that θ is indeed the
optimal solution of the quadratic optimization problem.
Theorem 1 Let ν∗
r = min{θ, Dr} where θ is the solution of k
r=1 minr{θ, Dr} = k
r=1 Dr−
1. Then, for every point ν¯ we have that ν¯2 > ν¯∗2.
Proof Assume by contradiction that there is another feasible point ¯ν = ¯ν∗ + ∆ which ¯
minimizes the objective function. Since ¯ν 
= ¯ν∗ we know that ∆¯ 
= 0. Both ¯ν and ¯ν∗ satisfy
the equality constraint of Eq. (34), thus 
r ∆r = 0. Also, both points satisfy the inequality
constraint of Eq. (34) thus ∆r ≤ 0 when ν∗
r = Dr. Combining the last two equations with
the assumption that ∆¯ 
= 0 we get that ∆s > 0 for some s with ν∗
s = θ. Using again the
equality 
r ∆r = 0 we have that there exists an index u with ∆u < 0. Let us denote by
 = min{|∆s|, |∆u|}. We now define a new feasible point ¯ν as follows. Let ν
s = νs − ,
ν
u = νu + , and ν
r = νr otherwise. We now show that the norm of ¯ν is smaller than the
norm of ¯ν. Since ¯ν and ¯ν differ only in their s and u coordinates, we have that,
ν¯
2 − ν¯2 = (ν
s)
2 + (ν
u)
2 − (νs)
2 − (νu)
2 .
Writing the values of ¯ν in terms of ¯ν and  we get,
ν¯
2 − ν¯2 = 2 ( − νs + νu) .
277
Crammer and Singer
From our construction of ¯ν we have that either νu −  = θ>νs or νu − >θ ≥ νs and
therefore we get νu − >νs. This implies that
ν¯
2 − ν¯2 < 0 ,
which is clearly a contradiction.
We now use the above characterization of the solution to derive a simple fixed-point
algorithm that finds θ. We use the simple identity min{θ, Dr} + max{θ, Dr} = θ + Dr and
replace the minimum function with the above sum in Eq. (37) and get,

k
r=1
[θ + Dr − max{θ, Dr}] = 
k
r=1
Dr − 1 ,
which amounts to,
θ∗ = 1
k



k
r=1
max{θ∗, Dr}

− 1
k . (38)
Let us define,
F(θ) = 1
k



k
r=1
max{θ, Dr}

− 1
k . (39)
Then, the optimal value θ∗ satisfies
θ∗ = F(θ∗) . (40)
Eq. (40) can be used for the following iterative algorithm. The algorithm starts with an
initial value of θ and then computes the next value using Eq. (39). It continues iterating
by substituting each new value of θ in F(·) and producing a series of values for θ. The
algorithm halts when a required accuracy is met, that is, when two successive values of θ
are close enough. A pseudo-code of the algorithm is given in Figure 3. The input to the
algorithm is the vector D¯, an initial suggestion for θ, and a required accuracy . We next
show that if θ1 ≤ maxr Dr then the algorithm does converge to the correct value of θ∗.
Theorem 2 Let θ∗ be the fixed point of Eq. (40) (θ∗ = F(θ∗)). Assume that θ1 ≤ maxr Dr
and let θl+1 = F(θl). Then for l ≥ 1
|θl+1 − θ∗|
|θl − θ∗| ≤ 1 − 1
k ,
where k is the number of classes.
Proof Assume without loss of generality that maxr Dr = D1 ≥ D2 ≥ ... ≥ Dk ≥ Dk+1
def
=
−∞. Also assume that θ∗ ∈ (Ds+1, Ds) and θl ∈ (Du+1, Du) where u, s ∈ {1, 2,...,k}.
278
Multiclass Kernel-based Vector Machines
FixedPointAlgorithm(D, θ,  ¯ )
Input D¯, θ1, .
Initialize l = 0.
Repeat
• l ← l + 1.
• θl+1 ← 1
k
k
r=1 max{θl, Dr}

− 1
k .
Until




θl − θl+1
θl




≤ .
Assign for r = 1,...,k: νr = min{θl+1, Dr}
Return: τ¯ = ¯ν − B¯
A .
Figure 3: The fixed-point algorithm for solving the reduced quadratic program.
Thus,
θl+1 = F(θl)
= 1
k



k
r=1
max{θl, Dr}

− 1
k
= 1
k
 
k
r=u+1
θl

+
1
k
u
r=1
Dr

− 1
k
=

1 − u
k

θl +
1
k
u
r=1
Dr − 1

. (41)
Note that if θl ≤ maxr Dr then θl+1 ≤ maxr Dr. Similarly,
θ∗ = F(θ∗) = 
1 − s
k

θ∗ +
1
k
s
r=1
Dr − 1

⇒ θ∗ = 1
s
s
r=1
Dr − 1

.
(42)
We now need to consider three cases depending on the relative order of s and u. The first
case is when u = s. In this case we get that,
|θl+1 − θ∗|
|θl − θ∗| = |

1 − s
k

θl + 1
k (
s
r=1 Dr − 1) − θ∗|
|θl − θ∗|
= |

1 − s
k

θl + s
k θ∗ − θ∗|
|θl − θ∗|
= 1 − s
k ≤ 1 − 1
k .
where the second equality follows from Eq. (42). The second case is where u>s. In this
case we get that for all r = s + 1,...,u :
θl ≤ Dr ≤ θ∗ . (43)
279
Crammer and Singer
Using Eq. (41) and Eq. (42) we get,
θl+1 =

1 − u
k

θl +
1
k
u
r=1
Dr − 1

=

1 − u
k

θl +
s
k
1
s
s
r=1
Dr − 1

+
1
k
 u
r=s+1
Dr

=

1 − u
k

θl +
s
k
θ∗ +
1
k
 u
r=s+1
Dr

.
Applying Eq. (43) we obtain,
θl+1 ≤

1 − u
k

θl +
s
k
θ∗ +
1
k
(u − s)θ∗
=

1 − u
k

θl +
u
k
θ∗ .
Since θl+1 is bounded by a convex combination of θl and θ∗, and θ∗ is larger than θl, then
θ∗ ≥ θl+1. We therefore finally get that,
|θl+1 − θ∗|
|θl − θ∗| = θ∗ − θl+1
θ∗ − θl
≤
θ∗ − 
1 − u
k

θl − u
k θ∗
θ∗ − θl
= 1 − u
k ≤ 1 − 1
k .
The last case, where u<s, is derived analogously to the second case, interchanging the
roles of u and s.
From the proof we see that the best convergence rate is obtained for large values of u. Thus,
a good feasible initialization for θ1 can be minr Dr. In this case
θ2 = F(θ1) = 1
k

k
r=1
Dr

− 1
k .
This gives a simple initialization of the algorithm which ensures that the initial rate of
convergence will be fast.
We are now ready to describe the complete implementation of the algorithm for learning
multiclass kernel machine. The algorithm gets a required accuracy parameter and the value
of β. It is initialized with ¯τi = 0 for all indices 1 ≤ i ≤ m. This value yields a simple
initialization of the variables Fi,r. On each iteration we compute from Fi,r the value ψi for
each example and choose the example index p for which ψp is the largest. We then call the
fixed-point algorithm which in turn finds an approximate solution to the reduced quadratic
optimization problem for the example indexed p. The fixed-point algorithm returns a set
of new values for ¯τp which triggers the update of Fi,r. This process is repeated until the
value ψi is smaller than  for all 1 ≤ i ≤ m. The pseudo-code of the algorithm is given in
Figure 4.
280
Multiclass Kernel-based Vector Machines
Input {(¯x1, y1),...,(¯xm, ym)}.
Initialize for i = 1,...,m:
• τ¯i = ¯0
• Fi,r = −βδr,yi (r = 1 ...k)
• Ai = K(¯xi, x¯i)
Repeat:
• Calculate for i = 1 ...m: ψi = maxr Fi,r − min r : τi,r<δyi,r
Fi,r
• Set: p = arg max{ψi}
• Set for r = 1 ...k : Dr = Fp,r
Ap − τp,r + δr,yp and θ = 1
k
k
r=1 Dr

− 1
k
• Call: ¯τ 
p =FixedPointAlgorithm(D, θ, / ¯ 2). (See Figure 3)
• Set: ∆¯τp = ¯τ 
p − τ¯p
• Update for i = 1 ...m and r = 1 ...k: Fi,r ← Fi,r + ∆τp,r K (¯xp, x¯i)
• Update: ¯τp ← τ¯
p
Until ψp < β
Output : H(¯x) = arg maxr

i
τi,rK (¯x, x¯i)

.
Figure 4: Basic algorithm for learning a multiclass, kernel-based, support vector machine
using KKT conditions for example selection.
7. Implementation details
We have discussed so far the underlying principal and algorithmic issues that arise in the
design of multiclass kernel-based vector machines. However, to make the learning algorithm
practical for large datasets we had to make several technical improvements to the baseline
implementation. While these improvements do not change the underlying design principals
they lead to a significant improvement in running time. We therefore devote this section to
a description of the implementation details. To compare the performance of the different
versions presented in this section we used the MNIST OCR dataset1. The MNIST dataset
contains 60, 000 training examples and 10, 000 test examples and thus can underscore significant implementation improvements. Before diving into the technical details we would like
to note that many of the techniques are by no means new and have been used in prior implementation of two-class support vector machines (see for instance Platt, 1998, Joachims,
1998, Collobert and Bengio, 2001). However, a few of our implementation improvements
build on the specific algorithmic design of multiclass kernel machines.
1. Available at http://www.research.att.com/˜yann/exdb/mnist/index.html
281
Crammer and Singer
10−4 10−3 10−2 10−1 100 0.80
0.85
0.90
0.95
1.00
1.05
1.10
1.15
1.20
1.25
1.30 x 104
epsilon
Run Time (sec)
10−4 10−3 10−2 10−1 100 1.40
1.45
1.50
1.55
1.60
1.65
1.70
1.75
1.80
1.85
1.90
epsilon
Test Error
Figure 5: The run time (left) and test error (right) as a function of required accuracy .
Our starting point and base-line implementation is the algorithm described in Figure 2
combined with the fixed-point algorithm for solving the reduced quadratic optimization
problem. In the base-line implementation we simply cycle through the examples for a
fixed number of iterations, solving the reduced optimization problem, ignoring the KKT
conditions for the examples. This scheme is very simple to implement and use. However, it
spends unnecessary time in the optimization of patterns which are correctly classified with
a large confidence. We use this scheme to illustrate the importance of the efficient example
selection described in the previous section. We now describe the different steps we took,
starting from the version described in Section 5.
Using KKT for example selection This is the algorithm described in Figure 4. For
each example i and label r we compute Fi,r. These variables are used to compute ψi as
described in Section 5. On each round we choose the example p for which ψp is the largest
and iterate the process until the value of ψi is smaller for a predefined accuracy denoted
by . It turns out that the choice of  is not crucial and a large range of values yield good
results. The larger  is the sooner we terminate the main loop of the algorithm. Therefore,
we would like to set  to a large value as long as the generalization performance is not
effected. In Figure 5 we show the running time and the test error as a function of . The
results show that a moderate value of  of 0.1 already yields good generalization. The
increase in running time when using smaller values for  is between %20 to %30. Thus, the
algorithm is rather robust to the actual choice of the accuracy parameter  so long as it is
not set to a value which is evidently too large.
Maintaining an active set The standard implementation described above scans the
entire training set and computes ψi for each example ¯xi in the set. However, if only a few
support patterns constitute the multiclass machine then the vector ¯τ is the zero vector for
many example. We thus partition the set of examples into two sets. The first, denoted
by A and called the active set, is composed of the set of examples that contribute to the
solution, that is, A = {i|τ¯i 
= ¯0}. The second set is simply its complement, Ac = {i|τ¯i = ¯0}.
During the course of the main loop we first search for an example to update from the set A.
Only if such an example does not exist, which can happen iff ∀i ∈ A,ψi < , we scan the
282
Multiclass Kernel-based Vector Machines
101 102
−10−2
−10−3
−10−4
Iteration
Objective function value
w/o cooling 
with cooling
Figure 6: The value of the objective function Q as a function of the number of iteration for
a fixed and variable scheduling of the accuracy parameter .
set Ac for an example p with ψp > . If such an example exists we remove it from Ac, add
it to A, and call the fixed-point algorithm with that example. This procedure spends most
of it time adjusting the weights of examples that constitute the active set and adds a new
example only when the active set is exhausted. A natural implication of this procedure is
that the support patterns can come only from the active set.
Cooling of the accuracy parameter The employment of an active set yields significant
reduction in running time. However, the scheme also forces the algorithm to keep updating
the vectors ¯τi for i ∈ A as long there is even a single example i for which ψi > . This may
result in minuscule changes and a slow decrease in Q once most examples in A have been
updated. In Figure 6 we plot in bold line the value of Q as a function of the number of
iterations when  is kept fixed. The line has a staircase-like shape. Careful examination of
the iterations in which there was a significant drop in Q revealed that these are the iterations
on which new examples were added to the active set. After each addition of a new example
numerous iterations are spent in adjust the weights ¯τi. To accelerate the process, especially
on early iterations during which we mostly add new examples to the active set, we use a
variable accuracy parameter, rather than a fixed accuracy. On early iterations the accuracy
value is set to a high value so that the algorithm will mostly add new examples to the active
set and spend only a small time on adjusting the weights of the support patterns. As the
number of iterations increases we decrease  and spend more time on adjusting the weights
of support patterns. The result is a smoother and more rapid decrease in Q which leads
to faster convergence of the algorithm. We refer to this process of gradually decreasing
283
Crammer and Singer
102 103 104
100
101
102
103
104
105
Size of Training Set
Run Time
1
2
3
4
5
Figure 7: Comparison of the run-time on the MNIST dataset of the different versions as a
function of the training-set size. Version 1 is the baseline implementation. Version
2 uses KKT conditions for selecting an example to update. Version 3 adds the
usage of an active set and cooling of . Version 4 adds caching of inner-products.
Finally, version 5 uses data structures for representing and using sparse inputs.
 as cooling. We tested the following cooling schemes (for t = 0, 1,...): (a) exponential:
(t) = 0 exp(−t); (b) linear: (t) = 0/(t + 1) (c) logarithmic: (t) = 0/ log10(t + 10). The
initial accuracy 0 was set to 0.999. We found that all of these cooling schemes improve the
rate of decrease in Q, especially the logarithmic scheme for which (t) is relatively large for
a long period and than decreases moderately. The dashed line in Figure 6 designate the
value of Q as a function of the number of iterations using a logarithmic cooling scheme for
. In the particular setting of the figure, cooling reduces the number of iterations, and thus
the running time, by an order of magnitude.
Caching Previous implementations of algorithms for support vector machines employ
a cache for saving expensive kernel-based inner-products (see for instance Platt, 1998,
Joachims, 1998, Collobert and Bengio, 2001). Indeed, one of the most expensive steps
in the algorithm is the evaluation of the kernel. Our scheme for maintaining a cache is as
follows. For small datasets we store in the cache all the kernel evaluations between each
example in the active set and all the examples in the training set. For large problems with
many support patterns (and thus a large active set) we use a least-recently-used (LRU)
scheme as a caching strategy. In this scheme, when the cache is full we replace least used
inner-products of an example with the inner-products of a new example. LRU caching is
also used in SVMlight (Joachims, 1998).
284
Multiclass Kernel-based Vector Machines
Name No. of No. of No. of No. of
Training Examples Test Examples Classes Attributes
satimage 4435 2000 6 36
shuttle 5000 9000 7 9
mnist 5000 10000 10 784
isolet 6238 1559 26 617
letter 5000 4000 26 16
vowel 528 462 11 10
glass 214 5-fold cval 7 9
Table 1: Description of the small databases used in the experiments.
Data-structures for sparse input instances Platt (1998) and others have observed
that when many components of the input vectors are zero, a significant saving of space
and time can be achieved using data-structures for sparse vectors and computing only the
products of the non-zero components in kernel evaluations. Our implementation uses linked
list for sparse vectors. Experimental evaluation we performed indicate that it is enough to
have 20% sparseness of the input instances to achieve a speedup in time and reduction in
memory over a non-sparse implementation.
To conclude this section we give in Figure 7a comparison of the run-time of the various technical improvements we outlined above. Each version that we plot include all of
the previous improvements. The running-time of the version that includes all the algorithmic and implementation improvements is two orders of magnitude faster than the baseline
implementation. It took the fastest version 3 hours and 25 minutes to train a multiclass
kernel-machine on the MNIST dataset using a Pentium III computer running at 600MHz
with 2Gb of physical memory. (The fastest version included two more technical improvements which are not discussed here but will be documented in the code that we will shortly
make available.) In comparison, Platt (1998) reports a training time of 8 hours and 10
minutes for learning a single classifier that discriminates one digit from the rest. Therefore,
it takes over 80 hours to train a set of 10 classifiers that constitute a multiclass predictor.
Platt’s results were obtained using a Pentium II computer running at 266MHz. While there
are many different factors that influence the running time, the running time results above
give an indication of the power of our approach for multiclass problems. We believe that
the advantage of our direct approach to multiclass problems will become even more evident
in problems with a large number of classes such as Kanji character recognition. We would
also like to note that the improved running time is not achieved at the expense of deteriorated classification accuracy. In the next section we describe experiments that show that
our approach is comparable to, and sometimes better than, the standard support vector
technology.
