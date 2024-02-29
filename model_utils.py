import sys
sys.path.append('graphcast')
from graphcast import checkpoint, graphcast, casting, normalization, autoregressive, data_utils, rollout
import dataclasses
import xarray as xr
import haiku as hk
import functools
import jax
import numpy as np
import time

class GraphCastModel:
    """
    A class to represent a GraphCast model for weather prediction.

    Attributes:
        number_of_predictions (int): The number of future predictions the model should make.
        params (dict): The parameters of the model.
        state (dict): The state of the model.
        model_config (dict): Configuration for the model.
        task_config (dict): Configuration for the task.
        diffs_stddev_by_level (xr.Dataset): Standard deviations of differences by level.
        mean_by_level (xr.Dataset): Mean values by level.
        stddev_by_level (xr.Dataset): Standard deviations by level.
        example_batch (dict): An example batch of data for evaluation.
    """

    def __init__(self, src_checkpoint: str, src_diffs_stddev_by_level: str, src_mean_by_level: str, src_stddev_by_level: str, example_batch: dict, number_of_predictions: int, new_input_duration: str = '18h'):
        """
        Constructs all the necessary attributes for the GraphCastModel object.

        Args:
            src_checkpoint (str): Source checkpoint path.
            src_diffs_stddev_by_level (str): Source path for standard deviations of differences by level.
            src_mean_by_level (str): Source path for mean values by level.
            src_stddev_by_level (str): Source path for standard deviations by level.
            example_batch (dict): An example batch of data for evaluation.
            number_of_predictions (int): The number of future predictions to make.
            new_input_duration (str): New input duration for the task configuration.
        """
        self.number_of_predictions = number_of_predictions
        # Load and modify checkpoint
        self.params, self.state, self.model_config, self.task_config = self.load_and_modify_checkpoint(src_checkpoint, new_input_duration)
        
        # Load datasets
        self.diffs_stddev_by_level, self.mean_by_level, self.stddev_by_level = self.load_datasets(src_diffs_stddev_by_level, src_mean_by_level, src_stddev_by_level)
        self.example_batch = example_batch

        # Prepare the model for running forward and gradients computation
        self.prepare_model()

    @staticmethod
    def load_and_compute_dataset(path: str) -> xr.Dataset:
        """
        Loads and computes a dataset from a given path.

        Args:
            path (str): The path to the dataset.

        Returns:
            xr.Dataset: The loaded and computed dataset.
        """
        with open(path, "rb") as f:
            dataset = xr.load_dataset(f).compute()
        return dataset

    def load_datasets(self, src_diffs_stddev_by_level: str, src_mean_by_level: str, src_stddev_by_level: str) -> tuple:
        """
        Loads datasets for standard deviations of differences, means, and standard deviations by level.

        Args:
            src_diffs_stddev_by_level (str): Source path for standard deviations of differences by level.
            src_mean_by_level (str): Source path for mean values by level.
            src_stddev_by_level (str): Source path for standard deviations by level.

        Returns:
            tuple: A tuple containing the loaded datasets for standard deviations of differences, means, and standard deviations by level.
        """
        return (
            self.load_and_compute_dataset(src_diffs_stddev_by_level),
            self.load_and_compute_dataset(src_mean_by_level),
            self.load_and_compute_dataset(src_stddev_by_level)
        )

    @staticmethod
    def load_and_modify_checkpoint(src: str, new_input_duration: str) -> tuple:
        """
        Loads and modifies a checkpoint from a given source.

        Args:
            src (str): The source path for the checkpoint.
            new_input_duration (str): The new input duration to set in the task configuration.

        Returns:
            tuple: A tuple containing the parameters, state, model configuration, and modified task configuration.
        """
        with open(src, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        
        task_config = dataclasses.replace(task_config, input_duration=new_input_duration)
        
        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")
        print()
        print()
        
        return params, state, model_config, task_config

    def drop_state(self, fn):
        """
        Drops the state from the function's output, returning only the first element of the output tuple.

        Args:
            fn (function): The function whose state should be dropped.

        Returns:
            function: A wrapped version of the input function that only returns the first element of its output.
        """
        return lambda **kw: fn(**kw)[0]

    def with_configs(self, fn):
        """
        Partially applies model and task configurations to the given function.

        Args:
            fn (function): The function to be partially applied.

        Returns:
            function: The input function with model_config and task_config partially applied.
        """
        return functools.partial(fn, model_config=self.model_config, task_config=self.task_config)

    def with_params(self, fn):
        """
        Partially applies model parameters and state to the given function.

        Args:
            fn (function): The function to be partially applied.

        Returns:
            function: The input function with params and state partially applied.
        """
        return functools.partial(fn, params=self.params, state=self.state)

    def construct_wrapped_graphcast(self):
        """
        Constructs the GraphCast predictor with necessary preprocessing and postprocessing steps.

        Returns:
            Predictor: The constructed GraphCast predictor.
        """
        predictor = graphcast.GraphCast(self.model_config, self.task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, self.diffs_stddev_by_level, self.mean_by_level, self.stddev_by_level)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    def run_forward(self, inputs, targets_template, forcings, model_config=None, task_config=None, params=None, state=None):
        """
        Runs a forward pass of the model.

        Args:
            inputs (array): The input data.
            targets_template (array): The template for targets.
            forcings (array): The forcing data.
            model_config (dict, optional): The model configuration. Defaults to None.
            task_config (dict, optional): The task configuration. Defaults to None.
            params (dict, optional): The model parameters. Defaults to None.
            state (dict, optional): The model state. Defaults to None.

        Returns:
            array: The model's predictions.
        """
        predictor = self.construct_wrapped_graphcast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def grads_fn(self, inputs, targets, forcings):
        """
        Computes the gradients of the loss with respect to the model parameters.

        Args:
            inputs (array): The input data.
            targets (array): The target data.
            forcings (array): The forcing data.

        Returns:
            tuple: A tuple containing the loss, diagnostics, next state, and gradients.
        """
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(params, state, jax.random.PRNGKey(0), self.model_config, self.task_config, i, t, f)
            return loss, (diagnostics, next_state)
        
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(self.params, self.state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    def get_eval_inputs_eval_targets_eval_forcings(self, example_batch):
        """
        Extracts evaluation inputs, targets, and forcings from the example batch.

        Args:
            example_batch (dict): An example batch of data.

        Returns:
            tuple: A tuple containing evaluation inputs, targets, and forcings.
        """
        eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
            example_batch, target_lead_times=slice("6h", f"{1*6}h"), **dataclasses.asdict(self.task_config)
        )
        return eval_inputs, eval_targets, eval_forcings
    
    def adjust_eval_targets_and_forcings_time_dim(self):
        """
        Adjusts the time dimensions of evaluation targets and forcings based on the number of predictions.
        """
        if self.number_of_predictions > 1:
            eval_targets_to_concat = [self.eval_targets*np.nan for _ in range(self.number_of_predictions)]
            self.eval_targets = xr.concat(eval_targets_to_concat, dim='time')

            eval_forcings_to_concat = [self.eval_forcings if i == 0 else self.eval_forcings*np.nan for i, _ in enumerate(range(self.number_of_predictions))]
            self.eval_forcings = xr.concat(eval_forcings_to_concat, dim='time')

            hours = np.arange(6, (self.eval_targets.time.size+1)*6, 6)
            nanoseconds = hours * 1e9 * 3600
            time_deltas = np.array(nanoseconds, dtype='timedelta64[ns]')

            if self.eval_forcings.time.size == time_deltas.size and self.eval_targets.time.size == time_deltas.size:
                self.eval_targets['time'] = time_deltas
                self.eval_forcings['time'] = time_deltas
                print("Time coordinates updated successfully.")
                print()
                print()
            else:
                raise ValueError("Size mismatch. Time coordinates not updated.")

    def prepare_model(self):
        """
        Prepares the model for making predictions by transforming, initializing, and jitting necessary functions.
        """
        self.run_forward_transformed = hk.transform_with_state(self.run_forward)
        self.init_jitted = jax.jit(self.with_configs(self.run_forward_transformed.init))
        self.run_forward_jitted = self.drop_state(self.with_params(jax.jit(self.with_configs(self.run_forward_transformed.apply))))
        self.eval_inputs, self.eval_targets, self.eval_forcings = self.get_eval_inputs_eval_targets_eval_forcings(self.example_batch)
        self.adjust_eval_targets_and_forcings_time_dim()

    def make_predictions(self):
        """
        Makes chunked predictions using the prepared model.

        Returns:
            array: The model's predictions.
        """
        start_time = time.time()
        predictions = rollout.chunked_prediction(
            self.run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=self.eval_inputs,
            targets_template=self.eval_targets * np.nan,
            forcings=self.eval_forcings,
            num_steps_per_chunk=1,
            verbose=True
        )
        print("Prediction time:", (time.time() - start_time ) / 60, " min")
        print()
        print()

        return predictions
