# AQA Miniproject: Expressivity of the generating set
Florian Velten (6344097)

## Global plan
### Week 17 (20-26 Apr)
- Literature and formulation of hypotheses and goals
- Setup of code repo

### Week 18 (27 Apr-3 May)
- Establish theory
- Code core of project (two feature maps, data generation, training, plots)
- Initial tests with two feature maps
- Reformulation of goals and hypotheses

### Week 19 (4-10 May)
- Generate, train and compare with three or four feature maps
- Write down observations and conclusions
- Read into regularisation
- Implement regularisation

### Week 20 (11-17 May)
- Write report of results until now
- Analyse regularisation
- Write about regularisation

**DEADLINE: Fr 15 May**


---
## Goals
1. Quantify (algebraically, if possible) how the Fourier coefficients depend on the optimisation parameters, for a specific encoding strategy based on the $e^{ixH}$-type feature maps and a specific measurement operator
2. Formulate and test hypotheses regarding the relative performance (expressivity and generalisation) of the same encoding strategy with different hyperparameter settings and different encoding Hamiltonians
3. Formulate and test hypotheses regarding the relative performance (expressivity and generalisation) of another feature map with the same overall encoding strategy



---
## Log
### Week 17
#### Mo 20 Apr
Repo has been set up only with the basics.
Planning has been made (in Google Calendar), though I'll probably be struggling to find time to really make this project work and go the distance of an 8.5, which I hereby declare to be my ultimate goal.

#### Tu 21 Apr
Reading [Effect of data encoding on the expressive power of variational quantum-machine-learning models](https://doi.org/10.1103/PhysRevA.103.032430).
They express the function $f(x) = \langle 0|U^\dagger(x,\theta)MU(x,\theta)|0 \rangle$ as a (truncated/partial) Fourier series
$$f(x) = \sum_{\omega\in\Omega} c_\omega e^{i\omega x}$$
for easy analysis.
They treat the parametrised circuits as arbitrary unitaries $W^{(1)},\dots,W^{(L+1)}$ and alternate these with $L$ encoding layers given by $S(x) = \exp(ixH)$ for some fixed $H$, so that
$$U(x,\theta) = W^{(L+1)}S(x)W^{(L)}S(x)\dots W^{(2)}S(x)W^{(1)}.$$
The set of frequencies $\Omega$ only depends on the spectrum of $H$.
If the spectrum only contains integers (or integer multiples of a single arbitrary frequency), the resulting Fourier-type sum is a truncated Fourier series, i.e. $\Omega = \{-K,-K+1,\dots,K-1,K\}$.

#### Th 23 Apr
The size $K$ of the spectrum is upper bounded by the qubit depth $d$ of the encoding gate and the number of repetitions $L$ as $K \le \frac{1}{2}d^{2L}-1$.
On the other hand, to control the Fourier coefficients freely, one needs $M \ge 2K+1$ real degrees of freedom, which is to say tunable parameters.

*Potential path of inquiry:*
1. Choose one parametrisation block, determine the way it affects the Fourier coefficients to make predictions. The number of parameters needs to be chosen well
2. Compare different encoding Hamiltonians:
   1. Two with the same spectrum, to confirm dependence on only the spectrum. Do this for $d=L=1$ (so different Hamiltonians with the same (effective) spectrum) and also for larger $d,L$ (and compare with values of $d$ and $L$ swapped)
   2. Different spectra (needs $d\ge 2$ and/or $L\ge 2$), to confirm that more frequencies means more expressivity
   3. Also generate data using some known function instead of randomly chosen parameters (make predictions based on expressivity vs. what is necessary for the type of function). Use functions generated as truncated Fourier series and otherwise
3. Add comparisons with non-linearity in the exponent (e.g. $e^{i \arccos(x) H}$), which should be more expressive. Make comments on periodicity of the $e^{ixH}$-type feature maps

*Prerequisites:*
- Choose loss function to train Adam
- Choose fixed measurement operator (but not important if encoding is followed by arbitrary parametrised unitary)

*Ways to quantify model performance:*
- Generalisation loss (generalisation performance); compare with upper bounds in paper
- Number of training steps to obtain a certain training loss (training speed)
- Training runtime (training speed)

Reading [Encoding-dependent generalization bounds for
parametrized quantum circuits](https://arxiv.org/pdf/2106.03880).
They look at generalisation performance of PQCs in light of an existing body of work in statistical learning theory.
The main point is to quantify the *generalisation bound*, which quantifies the gap between the training loss and the generalisation loss for different encoding architectures.
Such bounds would allow *structural risk minimisation*, which minimises not only the training loss, but the combination of the training loss and the generalisation bound as a function of the model complexity (quantified by hyperparameters such as depth, width, etc.).
A minimum in this combination represents the optimal model complexity for the task at hand.
The authors go on to analyse different encoding strategies and their associated generalisation bound scaling.

Formulated goals 1-3.



### Week 18
#### Th 30 Apr
Reading [Fourier Fingerprints of Ansatzes in Quantum Machine Learning](https://arxiv.org/pdf/2508.20868).
They take a more close look at the dependence of the Fourier coefficients on the Ansatzes.
It seems that Circuit 19 from the referenced paper ([Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://arxiv.org/pdf/1905.10876)) has a good performance for favourable simplicity and parameter scaling, so I will use it.

As for the observable, the simple *total magnetisation* $M = \sum_{i=1}^{n} Z_i$ seems like a decent choice.

Wrote initial code for the variational Ansatz and the observable.


#### Fr 1 May
Fixed some errors in the code from yesterday.

After some thinking (and chatting with copilot), I came to the conclusion that "different classes of generators of rotations" refer to types of Hamiltonians.
For example, different classes might be "single-qubit", or "k-local", or something richer.
I'll want to pick my classes (once I can compare two, I can compare any number) and select two specific elements per class.
One element should already be diagonal, the other should not.
Here they are:

| Class                                     | $\Omega$ (at $L=1$)                 | Element 1                            | Element 2                               |
|-------------------------------------------|-------------------------------------|--------------------------------------|-----------------------------------------|
| Single-qubit Pauli                        | $\{-1,0,+1\}$                       | $\frac{1}{2}Z_1$                     | $\frac{1}{2}Y_1$                        |
| Sum of 3 Paulis                           | $\{-3,\dots,+3\}$                   | $\frac{1}{2}(Z_1+Z_2+Z_3)$           | $\frac{1}{2}(X_1+Y_2+Z_3)$              |
| Engineerd spectrum                        | large subset of $\{-34,\dots,+34\}$ | $\mathrm{diag}(0,1,4,9,15,22,32,34)$ | omit if no difference from basis change |
| Random linear combination of three Paulis | random                              |                                      |                                         |

See [Golomb ruler](https://en.wikipedia.org/wiki/Golomb_ruler) for the engineered spectrum one.
The missing frequencies are 16, 20, 24, 26, 27, and 29.

I will want to restrict myself to a single data dimension, since that is not a constraint I'm required to vary.
One real input means simple visualisation and low complexity.

I can also vary the layer, which would enrich the spectrum, but I'll start with $L=1$.

One could also vary whether the spectrum consists of integers or not.
However, irrational values of the spectrum seem unphysical in some sense, or at least algebraically independent values seem unphysical.
Thus, since we can always rescale so that rational eigenvalues all become integers, I will limit myself to integers in the first place.

Next steps:
1. Analyse Fourier coefficient range per circuit
2. Generate random coefficients per class instance and thus function for the circuits to approximate
3. Generate fully random instance of function class (e.g. polynomial, truncated Fourier series)


Worked on analysing Fourier coefficients.
Turns out that `np.fft.fft(values, norm='forward')` does exactly what you want if you feed it the output values of the target function for inputs `np.linspace(0, 2*np.pi, n_points)`.
The value at index $k$ gives you the value of coefficient $c_k$.
All possible coefficients can be obtained as array using `np.fft.fftfreq(n_points, 1/n_points)`.

Wrote code to visualise the Fourier coefficient spread of the different generators.
Now that I write this, I realise though that this is not too interesting, since the spread is determined mostly by the ansatz, which I keep fixed.
I have however confirmed that the generators behave as expected: the single-qubit Z rotation only generates sinusoids, the sum of three Z rotations generates frequencies up to $\pm 3$, and the engineerd spectrum generator generates frequencies up to $\pm 34$.

Here's a way to interpret "using the encoding as regularisation technique": In theory, you'd want to maximise expressibility given the amount of qubits, so maximise the size of $\Omega$.
However, this could induce overfitting, so limiting the size of $\Omega$ could be interpreted as a regularisation technique.
A way to analyse this could be to design generators with different sizes of $\Omega$ and to then check whether undershooting, overshooting, or synchronising with the maximum frequency in the 

Ended with having done steps 1 and 2 above, which is technically enough to start fitting some data... next week.
