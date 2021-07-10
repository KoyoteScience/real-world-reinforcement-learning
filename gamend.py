from .local_bandito import LocalBanditoSequence
from .utils import none_if_error, other_if_error
import random, itertools
import numpy as np


class OsbandGame:

    def reset(self):
        self.state = self.start_state
        self.terminated = False
        self.episode_length = 0
        self.history_of_context_feature_vectors = []
        self.history_of_action_feature_vectors = []
        self.history_of_states = [self.state]

    def __init__(self,
                 dimension_labels=None,
                 start_state=None,
                 number_of_states=5,
                 allow_leftward_drift=True,
                 end_state=None,
                 horizon=None):
        '''
        Base class for RiverSwim-type games based on the Osband papers (see, e.g., https://arxiv.org/pdf/1306.0940.pdf)
        :param dimension_labels: labels for each dimension (generally ['x', 'y', 'z', etc.]
        :param start_state: state that each game starts at, as a dictionary (e.g., {'x': 0, 'y': 0})
        :param number_of_states: integer, in each dimension 
        :param allow_leftward_drift: boolean, whether the game randomly pushes you to the left ('x-minus')
        :param end_state: where the reward is, e.g., {'x': 3, 'y': 3}
        :param horizon: how many steps in each episode
        '''

        if dimension_labels is None:
            dimension_labels = ['x', 'y']
        self.dimension_labels = dimension_labels
        if start_state is None:
            start_state = {label: 0 for label in self.dimension_labels}
        self.start_state = start_state
        if end_state is None:
            end_state = {label: number_of_states - 1 for label in self.dimension_labels}
        self.available_states = []
        available_states_as_tuples = list(
            itertools.product(*[list(range(number_of_states))] * len(self.dimension_labels)))
        for state_as_tuple in available_states_as_tuples:
            self.available_states.append(
                {self.dimension_labels[i]: state_as_tuple[i] for i in range(len(state_as_tuple))})
        if horizon is None:
            horizon = (number_of_states - 1) * len(self.dimension_labels)
        self.number_of_states = number_of_states

        self.end_state = end_state
        self.state = start_state
        self.allow_leftward_drift = allow_leftward_drift
        self.terminated = False
        self.episode_length = 0
        self.map_action_and_context_to_total_reward = {}
        self.map_action_and_context_to_counts = {}
        self.map_state_to_action_to_total_reward = {}
        self.map_state_to_action_to_counts = {}
        self.history_of_context_feature_vectors = []
        self.history_of_action_feature_vectors = []
        self.history_of_states = [self.state]
        self.horizon = horizon
        self.terminated = False

        # note that in a more general game, all of the following values can change from state to state
        list_of_actions = []
        for label in self.dimension_labels:
            for direction in ['minus', 'plus']:
                list_of_actions.append(label + '-' + direction)
        self.action_feature_metadata = [
            {
                'categorical_or_continuous': 'categorical',
                'name': 'action',
                'possible_values': set(list_of_actions)
            }]
        self.list_of_action_feature_vectors_to_choose_from = [[x] for x in list_of_actions]

        self.reset()
        return

    def get_new_state_after_action(self, state, action):
        '''
        returns the new state after taking action
        :param state: state as dictionary mapping dimension label to integer
        :param action: action as an action feature vector (list)
        :return: next state as dictionary
        '''

        state = state.copy()
        label, direction = action[0].split('-')

        if label not in self.dimension_labels:
            raise ValueError(
                f'Incorrect action value ({action}) must be one of {self.list_of_action_feature_vectors_to_choose_from}')

        if self.terminated:
            raise ValueError('Game has already terminated')

        if direction == 'minus':
            if state[label] > 0:
                state[label] -= 1
        elif direction == 'plus':
            if state[label] < self.number_of_states - 1:
                state[label] += 1

        if self.allow_leftward_drift:
            if random.random() < (1.0 - 1.0 / self.number_of_states):
                if state['x'] > 0:
                    state['x'] -= 1

        return state

    @staticmethod
    def convert_dict_to_tuple(state):
        '''
        helper function since dictionaries are unhashbale and can't be used as keys
        :param state: state as dictionary
        :return: state as tuple
        '''
        return tuple(item for item in sorted(state.items(), key=lambda x: x[0]))

    def take_action_and_get_reward(self, action, context):
        '''
        Given an action feature vector and context feature vector, take that action, update the internal state
        and get a reward
        :param action: action feature vector 
        :param context: context feature vector
        :return: reward as float or int
        '''

        state_at_beginning = self.convert_dict_to_tuple(self.state.copy())
        action_and_context_at_beginning = (tuple(action), tuple(context))

        self.state = self.get_new_state_after_action(self.state, action)
        self.episode_length += 1
        reward = self.get_reward()
        terminal = self.get_terminal()
        self.terminated = terminal
        self.history_of_context_feature_vectors.append(context)
        self.history_of_action_feature_vectors.append(action)
        self.history_of_states.append(self.state)

        # TODO: Verbose dictionary updates. Should I use defaultdict?
        # If terminal state, backtrack
        if terminal:
            for (tmp_action, tmp_state) in zip(self.history_of_action_feature_vectors, self.history_of_states):
                tmp_state = self.convert_dict_to_tuple(tmp_state)
                if tmp_state not in self.map_state_to_action_to_total_reward:
                    self.map_state_to_action_to_total_reward[tmp_state] = {}
                if tuple(tmp_action) not in self.map_state_to_action_to_total_reward[tmp_state]:
                    self.map_state_to_action_to_total_reward[tmp_state][tuple(tmp_action)] = 0
                self.map_state_to_action_to_total_reward[tmp_state][tuple(tmp_action)] += reward
            for (tmp_action, tmp_context) in zip(self.history_of_action_feature_vectors,
                                                 self.history_of_context_feature_vectors):
                tmp_action_and_context = (tuple(tmp_action), tuple(tmp_context))
                if tmp_action_and_context not in self.map_action_and_context_to_total_reward:
                    self.map_action_and_context_to_total_reward[tmp_action_and_context] = 0
                self.map_action_and_context_to_total_reward[tmp_action_and_context] += reward

        if state_at_beginning not in self.map_state_to_action_to_counts:
            self.map_state_to_action_to_counts[state_at_beginning] = {}
        if tuple(action) not in self.map_state_to_action_to_counts[state_at_beginning]:
            self.map_state_to_action_to_counts[state_at_beginning][tuple(action)] = 0
        self.map_state_to_action_to_counts[state_at_beginning][tuple(action)] += 1

        if action_and_context_at_beginning not in self.map_action_and_context_to_counts:
            self.map_action_and_context_to_counts[action_and_context_at_beginning] = 0
        self.map_action_and_context_to_counts[action_and_context_at_beginning] += 1

        return reward

    def get_reward(self):
        '''
        given the internal state, what reward do we get?
        :return: reward as int or float
        '''
        reward = 0
        if self.state == self.end_state:
            reward = 1
        return reward

    def get_terminal(self):
        '''
        given the internal state, is it terminal?
        :return: boolean for whether state is terminal or we've exceeded the horizon
        '''

        reward = self.get_reward()
        terminal = False
        if reward == 1:
            terminal = True

        if self.episode_length >= self.horizon:
            terminal = True

        return terminal


class MasonGame(OsbandGame):

    def __init__(self, *args,
                 minimum_left_turns_for_reward=0,
                 list_of_states_where_left_turns_are_required_for_reward=None,
                 **kwargs):
        '''
        Extends OsbandGame to incorporate assumptions in real-world applications such as:
        1) State history determines reward, for example:
          a) a minimum number of left turns
          b) a list of states where the left turn was required
        :param args: catch basin for arguments that work for OsbandGame
        :param minimum_left_turns_for_reward: integer, number of left turns required 
          to get a reward at the reward state (1a)
        :param list_of_states_where_left_turns_are_required_for_reward: list of dictionaries 
          where a left turn is required to get a reward at the reward state (1b)
        :param kwargs: catch basin for keyword arguments that work for OsbandGame
        '''
        super().__init__(*args, **kwargs)

        if list_of_states_where_left_turns_are_required_for_reward is None:
            list_of_states_where_left_turns_are_required_for_reward = []
        self.list_of_states_where_left_turns_are_required_for_reward = \
            list_of_states_where_left_turns_are_required_for_reward

        minimum_left_turns_for_reward = max(minimum_left_turns_for_reward,
                                            len(self.list_of_states_where_left_turns_are_required_for_reward))
        self.minimum_left_turns_for_reward = minimum_left_turns_for_reward

    def get_reward(self):
        '''
        given the internal state, what reward do we get?
        :return: reward as int or float
        '''
        reward = 0

        # reward only occurs at the final state
        condition1 = self.state == self.end_state

        # add minimum number of left turns requirement
        condition2 = sum(
            1 if x == ['x-minus'] else 0 for x in self.history_of_action_feature_vectors
        ) >= self.minimum_left_turns_for_reward

        # add left turns at specific states
        condition3 = True

        for state_where_left_turn_is_required in self.list_of_states_where_left_turns_are_required_for_reward:
            sub_condition = False
            for state, action in zip(self.history_of_states, self.history_of_action_feature_vectors):
                if state == state_where_left_turn_is_required and action == ['x-minus']:
                    sub_condition = True
            if not sub_condition:
                condition3 = False

        if all([condition1, condition2, condition3]):
            reward = 1

        return reward


class PlayGameEpisodes:

    def __init__(self,
                 game=OsbandGame(),
                 observable_dimension_labels=None,
                 q_learning=True,
                 use_interaction_features=True,
                 resample_params_every_step=True,
                 same_bandit_for_every_step=True,
                 add_left_presence_to_context=False,
                 add_left_count_continuous_to_context=False,
                 add_left_count_categorical_to_context=False,
                 add_state_to_context=False,
                 add_step_number_to_context=False,
                 add_state_step_interaction_to_context=False,
                 strategy='greedy'
                 ):
        '''
        Class for managing game play for either OsbandGame or MasonGame using bandits.
        Note that this adds another assumption for real-world reinforcement learning from MasonGame:
          only a subset of the dimensions are observable!
        :param game: class instance of OsbandGame or MasonGame
        :param observable_dimension_labels: list of dimension labels that you can observe
        :param q_learning: boolean for whether we use Q-learning or train against summed future rewards
        :param use_interaction_features: boolean for whether the bandit uses action-context interaction terms
        :param resample_params_every_step: boolean for whether we resample our bandit model parameters for every step
        :param same_bandit_for_every_step: boolean for whether one bandit is used for each step or separate bandits are used
        :param add_left_presence_to_context: do we add a boolean for whether we have ever turned left to the context feature vector?
        :param add_left_count_continuous_to_context: do we add an integer for how many times we have turned left to the context feature vector?
        :param add_left_count_categorical_to_context: do we add a categorical variable for how many times we have turned left to the context feature vector?
        :param add_state_to_context: do we add the state (e.g., {'x': 0, 'y': 0} as a tuple) to the context feature vector as a categorical variable?
        :param add_step_number_to_context: do we add the number of steps so far as a categorical value to the context feature vetor
        :param add_state_step_interaction_to_context: do we add the interaction term between step and state to the context feature vector
        :param strategy: which strategy do we use? (generally just use 'greedy')
        '''
        self.game = game
        self.observable_dimension_labels = observable_dimension_labels
        self.episode_count = 0
        self.q_learning = q_learning
        self.use_interaction_features = use_interaction_features
        self.resample_params_every_step = resample_params_every_step
        self.same_bandit_for_every_step = same_bandit_for_every_step
        self.add_left_presence_to_context = add_left_presence_to_context
        self.add_left_count_continuous_to_context = add_left_count_continuous_to_context
        self.add_left_count_categorical_to_context = add_left_count_categorical_to_context
        self.add_state_to_context = add_state_to_context
        self.add_step_number_to_context = add_step_number_to_context
        self.add_state_step_interaction_to_context = add_state_step_interaction_to_context
        self.strategy = strategy
        self.agent = None
        self.total_reward = 0

        self.context_feature_metadata = []

        possible_states_as_tuples = [
            self.convert_dict_to_tuple_of_key_sorted_values(self.scrub_unobservable_dimension_labels(x)) for x in
            game.available_states]
        if self.add_state_to_context:
            self.context_feature_metadata += [
                {
                    'categorical_or_continuous': 'categorical',
                    'name': 'state',
                    'possible_values': possible_states_as_tuples,
                }]
        if self.add_step_number_to_context:
            self.context_feature_metadata += [
                {
                    'categorical_or_continuous': 'categorical',
                    'name': 'step_number',
                    'possible_values': list(range(self.game.horizon)),
                }]
        if self.add_state_step_interaction_to_context:
            self.context_feature_metadata += [
                {
                    'categorical_or_continuous': 'categorical',
                    'name': 'state_and_step_number',
                    'possible_values': [(x[0], x[1]) for x in itertools.product(possible_states_as_tuples,
                                                                                list(range(self.game.horizon)))],
                }]
        if self.add_left_presence_to_context:
            self.context_feature_metadata += [
                {
                    'categorical_or_continuous': 'categorical',
                    'name': 'left_action_ever_taken',
                    'possible_values': {True, False},
                }]
        if self.add_left_count_continuous_to_context:
            self.context_feature_metadata += [
                {
                    'categorical_or_continuous': 'continuous',
                    'name': 'number_of_left_actions_taken'
                }]
        if self.add_left_count_categorical_to_context:
            self.context_feature_metadata += [
                {
                    'categorical_or_continuous': 'categorical',
                    'name': 'number_of_left_actions_taken',
                    'possible_values': list(range(self.game.horizon))
                }]

        self.restart_training()
        return

    def scrub_unobservable_dimension_labels(self, input_dict):
        '''
        given a state as a dictionary, remove the unobservable dimensions
        :param input_dict: state dictionary with all dimensions
        :return: state dictionary with unobservable dimensions removed
        '''
        output_dict = input_dict.copy()
        for key in input_dict:
            if key not in self.observable_dimension_labels:
                del output_dict[key]
        return output_dict

    @staticmethod
    def convert_dict_to_tuple_of_key_sorted_values(input_dict):
        '''
        Convert the state dictionary to a tuple of key-sorted values so we can hash it
        :param input_dict: state dictionary
        :return: tuple of key-sorted values
        '''
        return tuple(input_dict[key] for key in sorted(input_dict.keys()))

    def get_context_feature_vector(self):
        '''
        Given the current state and the options of the class instance, generate a context feature vector
        :return: context feature vector as list
        '''
        context_feature_vector = []
        if self.add_state_to_context:
            context_feature_vector.append(self.convert_dict_to_tuple_of_key_sorted_values(
                self.scrub_unobservable_dimension_labels(self.game.state)))
        if self.add_step_number_to_context:
            context_feature_vector.append(self.game.episode_length)
        if self.add_state_step_interaction_to_context:
            context_feature_vector.append((self.convert_dict_to_tuple_of_key_sorted_values(
                self.scrub_unobservable_dimension_labels(self.game.state)), self.game.episode_length))
        if self.add_left_presence_to_context:
            if ['x-minus'] == self.game.history_of_action_feature_vectors[0]:
                context_feature_vector.append(True)
            else:
                context_feature_vector.append(False)
        if self.add_left_count_continuous_to_context:
            context_feature_vector.append(
                sum([1 if x == ['x-minus'] else 0 for x in self.game.history_of_action_feature_vectors]))
        if self.add_left_count_categorical_to_context:
            context_feature_vector.append(
                sum([1 if x == ['x-minus'] else 0 for x in self.game.history_of_action_feature_vectors]))
        return context_feature_vector

    def play_episode_and_return_total_reward(
            self,
            list_of_actions=None,
            print_steps=True,
            strategy=None,
            predict_on_all_models=False
    ):
        '''
        Play a full game episode and return the total reward from the episode
        :param list_of_actions: if not None, a list of action feature vectors that are prescribed to take
        :param print_steps: boolean for whether to print all the steps you take with diagnostic info
        :param strategy: generally 'greedy'
        :param predict_on_all_models: boolean for whether we predict scores for all models in the distribution
          this is less efficient but it allows us to compare how each action compares to each other
        :return: total reward over the course of the episode
        '''
        self.game.reset()

        if strategy is None:
            strategy = self.strategy

        print_strategy = strategy
        if list_of_actions is not None:
            print_strategy = 'pre-determined'

        sequence_index = 0
        total_reward = 0

        if self.same_bandit_for_every_step:
            self.agent.bandit.resample_frozen_params()
        else:
            for bandit in self.agent.bandit:
                bandit.resample_frozen_params()

        while not self.game.terminated and self.game.episode_length < self.game.horizon:

            if self.same_bandit_for_every_step:
                bandit = self.agent.bandit
            else:
                bandit = self.agent.bandit[sequence_index]

            if self.resample_params_every_step:
                bandit.resample_frozen_params()

            list_of_action_feature_vectors_to_choose_from = self.game.list_of_action_feature_vectors_to_choose_from
            context_feature_vector = self.get_context_feature_vector()

            # note that we want to run the prediction even if we have predetermined the action
            action_index_to_take = bandit.select(list_of_action_feature_vectors_to_choose_from,
                                                 context_feature_vector,
                                                 strategy=strategy, predict_on_all_models=predict_on_all_models)
            propensity = bandit.selection_metadata['overall_propensity']
            if list_of_actions is None:
                action_feature_vector_to_take = list_of_action_feature_vectors_to_choose_from[action_index_to_take]
            else:
                try:
                    action_feature_vector_to_take = list_of_actions[self.game.episode_length]
                except:
                    raise ValueError(
                        f'list_of_actions: {list_of_actions} self.game.episode_length={self.game.episode_length} self.game.horizon={self.game.horizon}')

                propensity = 1

            reward = self.game.take_action_and_get_reward(action_feature_vector_to_take, context_feature_vector)
            self.agent.train_step(
                action_feature_vector_to_take,
                context_feature_vector,
                reward,
                weight=1.0 / propensity
            )
            if print_steps:
                state = self.game.history_of_states[-2]
                state_as_tuple = self.game.convert_dict_to_tuple(state)
                print(f'Step #{self.game.episode_length}, State: {state}')
                print(f'Prediction scores:')
                for action_index in range(len(self.game.list_of_action_feature_vectors_to_choose_from)):
                    action_feature_vector = self.game.list_of_action_feature_vectors_to_choose_from[action_index]
                    prediction_score = bandit.prediction_by_action_index[action_index]
                    max_score = bandit.selection_metadata['selection_metadata_by_prediction_sample_index'][0][
                        'max_score']
                    mean_prediction_score = none_if_error(lambda: np.average(
                        bandit.prediction_distribution_by_action_index[action_index]))
                    std_prediction_score = none_if_error(lambda: np.sqrt(
                        np.var(bandit.prediction_distribution_by_action_index[action_index])))

                    min_prior_count = bandit.get_min_prior_counts(action_feature_vector,
                                                                  self.game.history_of_context_feature_vectors[-1])
                    test_action_and_context = (tuple(action_feature_vector),
                                               tuple(self.game.history_of_context_feature_vectors[-1]))
                    try:
                        state_and_action_counts = self.game.map_state_to_action_to_counts[state_as_tuple][
                            tuple(action_feature_vector)]
                    except:
                        state_and_action_counts = 0
                    try:
                        state_and_action_total_reward = self.game.map_state_to_action_to_total_reward[state_as_tuple][
                            tuple(action_feature_vector)]
                    except:
                        state_and_action_total_reward = 0
                    try:
                        context_and_action_counts = self.game.map_action_and_context_to_counts[test_action_and_context]
                    except:
                        context_and_action_counts = 0
                    try:
                        context_and_action_total_reward = self.game.map_action_and_context_to_total_reward[
                            test_action_and_context]
                    except:
                        context_and_action_total_reward = 0
                    taken_addendum = '  '
                    if action_index == action_index_to_take:
                        taken_addendum = '--> '
                    print(
                        f'{taken_addendum}Action: {action_feature_vector}, Prediction Score: {other_if_error(lambda: prediction_score, lambda x: f"{x:.4f}")} ({other_if_error(lambda: mean_prediction_score - std_prediction_score, lambda x: f"{x:.4f}")} < {other_if_error(lambda: mean_prediction_score, lambda x: f"{x:.4f}")} < {other_if_error(lambda: mean_prediction_score + std_prediction_score, lambda x: f"{x:.4f}")})')
                    print(
                        f'    Total Reward for State + Action: {state_and_action_total_reward} out of {state_and_action_counts}')
                    print(
                        f'    Total Reward for Context + Action: {context_and_action_total_reward} out of {context_and_action_counts} (Min. Prior Count: {min_prior_count})')
                print(f'Context:')
                for context_feature_index in range(len(self.context_feature_metadata)):
                    print(
                        f'  Name: {self.context_feature_metadata[context_feature_index]["name"]} ({self.context_feature_metadata[context_feature_index]["categorical_or_continuous"]}), Value: {context_feature_vector[context_feature_index]}')
                print(
                    f'Took Action {action_feature_vector_to_take} ({print_strategy}), Now in State {self.game.state}, Got Reward {reward}')
                if action_feature_vector_to_take != ['x-minus'] and self.game.state['x'] < \
                        self.game.history_of_states[-2]['x'] or action_feature_vector_to_take == ['x-plus'] and \
                        self.game.state['x'] == 0:
                    print('  (drifted left)')
                if max_score is None:
                    print(f'  Explored!')
                else:
                    print(f'  Max Score: {max_score}')
                if prediction_score != max_score and print_strategy == 'greedy':
                    if not bandit.selection_metadata['should_we_assign_unknown_score_by_action_index']:
                        raise ValueError('Max score should match chosen action score for greedy strategy')

            total_reward += reward
            sequence_index += 1

        self.agent.train_sequence()

        if print_steps:
            print(f'Finished Episode #{self.episode_count + 1} with total reward {total_reward}\n')

        self.episode_count += 1
        self.total_reward += total_reward

        return total_reward

    def play_game_until_reward(self, max_episodes=100, print_steps=True):
        '''
        Play multiple game episodes until you get a non-zero total reward
        :param max_episodes: maximum number of episodes to try
        :param print_steps: boolean for printing out each step taken with diagnostic info
        :return: None
        '''
        total_reward = 0
        episode_length_at_beginning = self.episode_count

        while total_reward == 0:
            total_reward = self.play_episode_and_return_total_reward(print_steps=print_steps)
            if self.episode_count > max_episodes + episode_length_at_beginning:
                break

        if print_steps:
            if total_reward > 0:
                print(f'Got reward at episode: {self.episode_count}')
            else:
                print(f'Timed out after {self.episode_count} episodes')

        return

    def restart_training(self):
        '''
        Restart the bandits used to play the game
        :return: None
        '''
        self.game.reset()
        self.episode_count = 0
        try:
            horizon = self.game.horizon
        except:
            horizon = None

        # note that in a more general game, we can have a mapping from game states to agents
        #   but here it's just constant
        self.agent = LocalBanditoSequence(
            self.game.action_feature_metadata,
            self.context_feature_metadata,
            gamma=1,
            list_of_action_feature_vectors_to_choose_from=self.game.list_of_action_feature_vectors_to_choose_from,
            same_bandit_for_every_step=self.same_bandit_for_every_step,
            horizon=horizon,
            q_learning=self.q_learning,
            use_interaction_features=self.use_interaction_features,
            bandito_options={'trailing_list_length': 1000},
            strategy=self.strategy
        )


class EvaluateGamePerformance:

    # note that defaults values should be tuples instead of lists according to PyCharm
    def __init__(self,
                 dimension_labels=('x', 'y'),
                 observable_dimension_labels=('x',),
                 number_of_states=3,
                 list_of_states_where_left_turns_are_required_for_reward=({'x': 0, 'y': 0},),
                 list_of_steps_where_left_turns_are_required_for_reward=(0,),  # this has to be done manually for now
                 minimum_left_turns_for_reward=0,
                 allow_leftward_drift=False,
                 use_interaction_features=True,
                 same_bandit_for_every_step=False,
                 add_state_to_context=True,
                 add_step_number_to_context=True,
                 add_state_step_interaction_to_context=False,
                 add_left_presence_to_context=False,
                 add_left_count_continuous_to_context=False,
                 add_left_count_categorical_to_context=False,
                 q_learning=True,
                 number_of_priming_episodes=10,
                 number_of_self_training_episodes=100
                 ):
        '''
        Wrapper class for an entire bandit training sequence consisting of the following steps:
        1) Prime the bandits with successful runs
        2) Self-training for number_of_episodes episodes
        3) Print out the coefficients of the bandit models
        4) Print out diagnostics for each step for two games: 
           a) one where the actions are guaranteed to generate a reward
           b) one where the actions are not pretermined, to see if we learned waht we wanted
        :param dimension_labels: list of strings of dimension labels (['x', 'y', 'z', etc.])
        :param observable_dimension_labels: subset of dimension_labels that we can observe
        :param number_of_states: number of states in each dimension
        :param list_of_states_where_left_turns_are_required_for_reward: 
            list of state dictionaries where a left turn is required to get a reward at the reward state
        :param list_of_steps_where_left_turns_are_required_for_reward: 
            list of step integers where a left turn is required to get a reward at the reward state
        :param minimum_left_turns_for_reward: integer of number of steps required to obtain a reward at the reward state
        :param allow_leftward_drift: boolean for whether game randomly pushes you left ('x-minus')
        :param use_interaction_features: boolean for whether action-context interaction features are used in bandits
        :param same_bandit_for_every_step: boolean for whether the same bandit is used for each step
        :param add_state_to_context: do we add the state (e.g., {'x': 0, 'y': 0} as a tuple) to the context feature vector as a categorical variable?
        :param add_step_number_to_context: do we add the number of steps so far as a categorical value to the context feature vetor
        :param add_state_step_interaction_to_context: do we add the interaction term between step and state to the context feature vector
        :param add_left_presence_to_context: do we add a boolean for whether we have ever turned left to the context feature vector?
        :param add_left_count_continuous_to_context: do we add an integer for how many times we have turned left to the context feature vector?
        :param add_left_count_categorical_to_context: do we add a categorical variable for how many times we have turned left to the context feature vector?
        :param q_learning: boolean for whether we use Q-learning or train against summed future rewards
        :param number_of_priming_episodes: integer number of episodes we train with successful action sequence
        :param number_of_self_training_episodes: integer number of episodes where bandits self-train
        '''

        # TODO: I guess it's better to write this out explicitly but that's so annoying
        kwargs = locals().copy()
        for key, val in kwargs.items():
            setattr(self, key, val)

        number_of_steps = (number_of_states - 1) * len(dimension_labels)
        for state_where_left_turn_is_required in list_of_states_where_left_turns_are_required_for_reward:
            if state_where_left_turn_is_required['x'] == 0:
                number_of_steps += 1
            else:
                number_of_steps += 2

        self.number_of_steps = number_of_steps

        if not self.same_bandit_for_every_step:
            self.add_state_step_interaction_to_context = False

        self.game = MasonGame(
            number_of_states=self.number_of_states,
            dimension_labels=self.dimension_labels,
            allow_leftward_drift=self.allow_leftward_drift,
            minimum_left_turns_for_reward=self.minimum_left_turns_for_reward,
            list_of_states_where_left_turns_are_required_for_reward=self.list_of_states_where_left_turns_are_required_for_reward,
            horizon=self.number_of_steps
        )

        self.episode_player = PlayGameEpisodes(
            self.game,
            observable_dimension_labels=self.observable_dimension_labels,
            q_learning=self.q_learning,
            use_interaction_features=self.use_interaction_features,
            resample_params_every_step=True,
            same_bandit_for_every_step=self.same_bandit_for_every_step,
            add_state_to_context=self.add_state_to_context,
            add_step_number_to_context=self.add_step_number_to_context,
            add_state_step_interaction_to_context=self.add_state_step_interaction_to_context,
            add_left_presence_to_context=self.add_left_presence_to_context,
            add_left_count_continuous_to_context=self.add_left_count_continuous_to_context,
            add_left_count_categorical_to_context=self.add_left_count_categorical_to_context,
            strategy='greedy'
        )

        correct_list_of_actions = [['x-plus'] for _ in range(self.number_of_steps)]
        for step_index in self.list_of_steps_where_left_turns_are_required_for_reward:
            correct_list_of_actions[step_index] = ['x-minus']
        num_steps_right = 0
        for step_index, action in enumerate(correct_list_of_actions):
            if action == ['x-plus'] and step_index != 0:
                num_steps_right += 1
            if action == ['x-minus']:
                num_steps_right -= 1
            if num_steps_right >= self.number_of_states - 1:
                correct_list_of_actions[step_index] = ['y-plus']

        self.correct_list_of_actions = correct_list_of_actions
        self.episode_player.restart_training()

    def run_training(self):
        '''
        Run the training sequence described in the notes for the __init__ method
        :return: None
        '''

        # priming with correct action sequence
        for episode_index in range(self.number_of_priming_episodes):
            print(f'Priming Episode #: {episode_index + 1} out of {self.number_of_priming_episodes}')
            self.episode_player.play_episode_and_return_total_reward(list_of_actions=self.correct_list_of_actions,
                                                                     print_steps=False)

        for episode_index in range(self.number_of_self_training_episodes):
            print(f'Self-Training Episode #: {episode_index + 1} out of {self.number_of_self_training_episodes}')
            self.episode_player.play_episode_and_return_total_reward(print_steps=False)

        self.episode_player.agent.pretty_print_coefficients()
        
        self.episode_player.play_episode_and_return_total_reward(list_of_actions=self.correct_list_of_actions,
                                                                 predict_on_all_models=True,
                                                                 print_steps=True)

        self.episode_player.play_episode_and_return_total_reward(predict_on_all_models=True, print_steps=True)

    def get_average_total_reward_over_trials(
            self,
            number_of_restarts=100,
            number_of_episodes=100
    ):
        '''
        For a given number of training restarts and episodes of self-training, get average reward over the restarts
        :param number_of_restarts: integer
        :param number_of_episodes: integer
        :return: average reward over the restarts
        '''
        list_of_total_rewards = []
        for restart_index in range(number_of_restarts):
            self.episode_player.restart_training()
            try:
                total_reward = 0
                for _ in range(number_of_episodes):
                    total_reward += self.episode_player.play_episode_and_return_total_reward(print_steps=False)
                list_of_total_rewards.append(total_reward)
                print(f'Total reward for restart {restart_index}: {total_reward}')
            except Exception as e:
                print(e)
        final_mean = np.mean(list_of_total_rewards)
        print(f'Avg. total reward over {number_of_restarts} trials: {final_mean}')
        return final_mean
