# Post Training
- Pretrained Large Language Models are not necessarily *helpful* after training
  - e.g. Prompt: `Explain the moon landing to a 6 year old in a few sentences.` Result: `Explain the theory of gravity to a 6 year old.`
- **Model Alignment** adjusts LLMs to better align for *human needs* - two common methods for doing so are **instruction tuning** and **preference alignment**
  - The steps involved in aligning a base model is known as **post-training**
## Instruction Tuning
- With instruction tuning, the model is trained on a dataset of instruction-output pairs so that it can actually *learn* how to perform various tasks as well as how to follow instructions in general
    - Example tasks include translation, summarization, code generation, and so forth
- Instruction tuning is supervised, and typically involves the same language modeling objective used to train the original model (guess the next token)
- There exist many instruction-tuning datasets - e.g. Aya, Flan, etc.
  - Creation of Datasets:
    - One way is for instruction-output pairs to be written directly by humans, often by volunteers
      - This is quite costly and time consuming
    - Another way is to leverage existing datasets of supervised training data for various other natural language tasks - such as using translation/summarization task datasets and converting them into instruction prompts via templates
    - Yet another way is to automatically generate instruction-tuning based on existing annotation guidelines fed as prompts to LLMs
## Optimizing for Human Preferences
- **Preference-based learning** uses preference judgements to further improve the performance of finetuned LLMs, leveraging human *preferences* about the end output
- Preference data typically involves a set of outputs $o$ that have been sampled using $x$ as a prompt; if $o_i$ is preferred to $o_j$ then $o_i \succ o_j | x$
  - This data can be obtained in various ways: direct annotations of pairs of sampled outputs by trained annotators, implicit preference judgement from web resources (e.g. upvotes on Reddit), or preference *directly from LLMs*
- Mathematically, preferences can be modeled as probabilities to better illustrate the *degree* by which one output is preferred over the other
  - **Bradley-Terry Model**: $P(o_i \succ o_j | x) = \frac{1}{1 + e^{-(z_i - z_j)}} = \sigma(z_i - z_j)$
- A model should learn some sort of function $r(x, o)$ that assigns a scalar *reward* to the prompt/output pairs
  - $P(o_i \succ o_j | x) = \sigma(r(x, o_i) - r(x, o_j))$
  - This can be learned via gradient descent to minimize binary cross-entropy loss, given that ground truth $(o_i \succ o_j | x)$ indicates that $P(o_i \succ o_j | x) = 1$ and otherwise $P(o_i \succ o_j | x) = 0$
    - $L_{CE}(x, o_w, o_l) = -\log \sigma (r(x, o_w) - r(x, o_l))$, where $o_w$ is the winner and $o_l$ is the loser
    - $L_{CE} = -\mathbb{E}_{x, o_w, o_l \sim \mathcal{D}} [\log \sigma (r(x, o_w) - r(x, o_l))] $
  - Current approaches initialize a reward model from an existing pretrained LLM, replacing the language modeling head from teh final layer with a dense linear layer, and performing gradient descent to learn a good scoring function
### Preference-Based Learning
- The alignment process makes use of reinforcement learning
  - Find optimal policy $\pi^* = \argmax_{\pi_\theta} \mathbb{E}_{x \sim \mathcal D, o \sim \pi_\theta (o | x)}[r(x, o)]$
  - Goal is to find a policy (model) that maximizes the expected reward
    - The reward maximized is not the true reward from the environment, but rather a noisy surrogate for the true reward model ($r$)
  - The policy (model) does not start from scratch but rather starts from an already instruction tuned model
- To avoid the pretrained LLM forgetting its pretraining knowledge, the reward function includes a penalty from diverging too far from the starting point
  - $\pi^* = \argmax_{\pi_\theta} \mathbb{E}_{x \sim \mathcal D, o \sim \pi_\theta (o | x)}[r(x, o) - \beta \mathbb{D}_{KL}[\pi_\theta(o|x) || \pi_{\text{ref}}(o|x)]] = \argmax_{\pi_\theta} \mathbb{E}_{x \sim \mathcal D, o \sim \pi_\theta (o | x)} [r_{\phi(x, o)} - \beta \log \frac{\pi_\theta(o|x)}{\pi_{\text{ref}}(o|x)}]$
- **Proximal Policy Optimization (PPO)**: An on-policy reinforcement learning algorithm that uses an explicit reward model to iteratively improve the policy (LLM) while constraining each update to remain close to the previous policy
  - PPO optimizes the policy $\pi_\theta$ using the learned reward model $r_\phi(x, o)$ from preference data
  - **Core Objective**: The policy is updated to maximize the clipped surrogate objective:
    - $L^{\text{CLIP}}(\theta) = \mathbb{E}_{x, o} [\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$
    - Where $r_t(\theta) = \frac{\pi_\theta(o|x)}{\pi_{\theta_{\text{old}}}(o|x)}$ is the probability ratio between the new and old policy
    - $\hat{A}_t$ is the advantage estimate: $\hat{A}_t = r_\phi(x, o) - \beta \log \frac{\pi_\theta(o|x)}{\pi_{\text{ref}}(o|x)} - V(x)$
    - $V(x)$ is a learned value function estimating the expected return from state $x$
    - $\epsilon$ is a hyperparameter (typically 0.1 or 0.2) that clips the ratio to prevent large policy updates
  - **Clipping Mechanism**: The clipping ensures that the policy update is conservative
    - If the advantage is positive ($\hat{A}_t > 0$), the ratio is clipped to at most $1 + \epsilon$, preventing the new policy from being too optimistic
    - If the advantage is negative ($\hat{A}_t < 0$), the ratio is clipped to at least $1 - \epsilon$, preventing the new policy from being too pessimistic
  - **Training Process**:
    1. Sample prompts $x$ from the dataset $\mathcal{D}$ and generate outputs $o$ using the current policy $\pi_\theta$
    2. Compute rewards using the trained reward model: $r = r_\phi(x, o) - \beta \log \frac{\pi_\theta(o|x)}{\pi_{\text{ref}}(o|x)}$
    3. Compute advantages $\hat{A}_t$ using Generalized Advantage Estimation (GAE) or other methods
    4. Update the policy using multiple epochs of minibatch gradient ascent on $L^{\text{CLIP}}$
    5. Update the value function $V(x)$ to minimize: $L^{V}(\theta) = \mathbb{E}_x[(V_\theta(x) - V^{\text{target}})^2]$
  - **Advantages**: PPO is stable and sample-efficient compared to vanilla policy gradient methods; the clipping prevents catastrophic policy updates
  - **Disadvantages**: Requires training and maintaining both a reward model and a value function; computationally expensive as it requires multiple forward passes during training; can be unstable if hyperparameters are not tuned properly
- **Direct Preference Optimization (DPO)**: Gradient-based learning can be used to optimize an LLM without learning an explicit reward model
  - Given the aforementioned optimal policy, it can be shown that:
    - $r(x, o) = \beta \log \frac{\pi_\theta(o|x)}{\pi_{\text{ref}}(o|x)} + \beta \log Z(x)$
    - $Z(x) = \sum_y \pi_{\text{ref}}(o|x) \exp(\frac{1}{\beta} r(x, o))$
      - This is the partition function, which sums over all possible outputs $o$ given a prompt $x$
      - In practice, this cannot be computed
    - Given the Bradley-Terry model, the partition function can be 'cancelled':
      - $P(o_i \succ o_j | x) = \sigma(r(x, o_i) - r(x, o_j)) = \sigma(\beta \log \frac{\pi_\theta(o_i|x)}{\pi_{\text{ref}}(o_i|x)} - \beta \log \frac{\pi_\theta(o_j|x)}{\pi_{\text{ref}}(o_j|x)})$
    - This means that the likelihood of a preference pair can be expressed in terms of the two LLM policies rather than in terms of an explicit reward model, so:
      - $L_{\text{DPO}}(x, o_w, o_l) = -\log \sigma (\beta \log \frac{\pi_\theta(o_w|x)}{\pi_{\text{ref}}(o_w|x)} - \beta \log \frac{\pi_\theta(o_l|x)}{\pi_{\text{ref}}(o_l|x)})$
      - $L_{\text{DPO}} = -\mathbb{E}_{(x, o_w, o_l) \sim \mathcal{D}} [\log \sigma(\beta \log \frac{\pi_\theta(o_w|x)}{\pi_{\text{ref}}(o_w|x)} - \beta \log \frac{\pi_\theta(o_l|x)}{\pi_{\text{ref}}(o_l|x)})]$
  - Thus, with DPO, there is no need to train an explicit reward model