"""String edit distance learning.

After:

    Ristad, E. S.and Yianilos, P. N. 1998. Learning string-edit distance. IEEE
    Transactions on Pattern Analysis and Machine Intelligence 20(5): 522-532.
"""


from __future__ import annotations

import abc
import dataclasses
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy

import tqdm
from scipy import special

from . import actions, util

LARGE_NEG_CONST = -1e6


class SEDParameterError(Exception):
    """Error for when SED Parameters violate formula conditions."""

    pass


class SEDActionError(Exception):
    """Error for when unavailable edit action is passed."""

    pass


@dataclasses.dataclass
class ParamDict:
    """Class to maintain parameter weights.

    Maintains dictionary of substitution, deletion, and insertion
    action-probabilility pairs, along with float rerpresenting
    probability of end-of-string insertion. Assumes probabilities
    are logarithmic.

    Args:
        delta_sub (Dict[Tuple[Any, Any], float]): Substitution
            action-probability pairs. Key is tuple of
            (orignal symbol, new symbol), value is log-probabilities
            of substitution.
        delta_del (Dict[Any, float]): Deletion action-probability
            pairs. Key is deleted symbol,
            value is log-probabilities of action.
        delta_ins (Dict[Any, float]): Insertion action-probability
            pairs. Key is inserted symbol,
            value is log-probabilities of action.
        delta_eoc (float): Log-probability of end of string.
    """

    delta_sub: Dict[Tuple[Any, Any], float]
    delta_del: Dict[Any, float]
    delta_ins: Dict[Any, float]
    delta_eos: float

    def sum(self) -> float:
        """Sums all probabilities."""
        values = [self.delta_eos]
        for vs in (self.delta_sub, self.delta_ins, self.delta_del):
            values.extend(vs.values())
        # Uses sum of exponentiation to maintain logarithmic values.
        return special.logsumexp(values)

    @classmethod
    def from_params(cls, other: ParamDict) -> ParamDict:
        return cls(
            delta_sub=other.delta_sub.copy(),
            delta_del=other.delta_del.copy(),
            delta_ins=other.delta_ins.copy(),
            delta_eos=other.delta_eos,
        )

    def update_params(self, other: ParamDict) -> None:
        self.delta_sub = other.delta_sub.copy()
        self.delta_del = other.delta_del.copy()
        self.delta_ins = other.delta_ins.copy()
        self.delta_eos = other.delta_eos

    def write_params(self, filepath: str) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def read_params(cls, filepath: str) -> ParamDict:
        with open(filepath, "wb") as file:
            return pickle.load(file)


class StochasticEditDistance(abc.ABC):
    params: ParamDict

    def __init__(self, params):
        """SED model.

        Uses character edits (insertions, deletions, substitutions)
        to convert strings. Edit weights are learned with
        expectation-maximization.

        Args:
            params (ParamDict): SED log weights.
        """
        self.params = params
        # Default value for 0-probability unseen pairs.
        self.default = LARGE_NEG_CONST
        psum = self.params.sum()
        if not numpy.isclose(0.0, psum):
            raise SEDParameterError(f"Parameters do not sum to 1: {psum:.4f}")

    @classmethod
    def build_sed(
        cls,
        source_alphabet: Iterable[Any],
        target_alphabet: Iterable[Any],
        copy_probability: Optional[float] = 0.9,
    ) -> StochasticEditDistance:
        """Builds a SED given a source and a target alphabets.

        If copy_probability is not None, distributes this probability
        mass across all copy actions to bias SED towards copying.

        Args:
            source_alphabet (Iterable[Any]): symbols of all input strings.
            target_alphabet (Iterable[Any]): symbols of all target strings.
            copy_probability (Optional[float]): on weight init, how much mass
                to give to copy edits.

        Returns:
            StochasticEditDistance.
        """
        source_alphabet = frozenset(source_alphabet)
        target_alphabet = frozenset(target_alphabet)
        n = (
            len(source_alphabet) * len(target_alphabet)  # All sub.
            + len(source_alphabet)  # All deletions.
            + len(target_alphabet)  # All insertions.
            + 1  # End symb.
        )
        if copy_probability is None:
            uniform_weight = numpy.log(1 / n)
            log_copy_prob = uniform_weight  # Probability of a copy action.
            log_rest_prob = uniform_weight  # Probability of any other action.
        elif 0 < copy_probability < 1:
            # Splits copy mass over individual copy actions.
            num_copy_edits = len(target_alphabet & source_alphabet)
            num_rest = n - num_copy_edits  # Removes copy actions from sub.
            log_copy_prob = numpy.log(copy_probability / num_copy_edits)
            log_rest_prob = numpy.log((1 - copy_probability) / num_rest)
        else:
            raise SEDParameterError(
                f"0 < copy probability (= {copy_probability:.4f}) < 1"
                "does not hold"
            )
        delta_sub = {
            (s, t): log_copy_prob if s == t else log_rest_prob
            for s in source_alphabet
            for t in target_alphabet
        }
        delta_del = {s: log_rest_prob for s in source_alphabet}
        delta_ins = {t: log_rest_prob for t in target_alphabet}
        delta_eos = log_rest_prob
        params = ParamDict(delta_sub, delta_del, delta_ins, delta_eos)
        return cls(params)

    @classmethod
    def fit_from_data(
        cls,
        lines: Iterable[Tuple[Any]],
        copy_probability: Optional[float] = None,
        epochs: int = 10,
    ) -> StochasticEditDistance:
        """Fits StochasticEditDistance parameters from data.

        Args:
            lines (Iterable[Tuple[Any]]): source and target strings.
            copy_probability (Optional[float]): default probability mass for
                copy edits.
            epochs (int): number of EM epochs.

        Returns:
            StochasticEditDistance.
        """
        source_alphabet = set()
        target_alphabet = set()
        sources = []
        targets = []
        for line in lines:
            # Split lines manually to ignore features from yoyodyne.
            s = line[0]
            t = line[1]
            source_alphabet.update(s)
            target_alphabet.update(t)
            sources.append(s)
            targets.append(t)
        sed = cls.build_sed(source_alphabet, target_alphabet, copy_probability)
        util.log_info(f"Performing {epochs} epochs of EM")
        sed.em(sources, targets, epochs)
        return sed

    def forward_evaluate(
        self, source: Sequence[Any], target: Sequence[Any]
    ) -> numpy.ndarray:
        """Computes forward probabilities.

        Computes dynamic programming table (in log real) filled with forward
        log probabilities.

        Args:
            source (Sequence[Any]): source string.
            target (Sequence[Any]): target string.

        Returns:
            numpy.ndarray: forward log probability alignments.
        """
        T = len(source)
        V = len(target)
        alpha = numpy.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.0
        for t in range(T + 1):
            for v in range(V + 1):
                summands = [alpha[t, v]]
                if v > 0:
                    summands.append(
                        self.params.delta_ins.get(target[v - 1], self.default)
                        + alpha[t, v - 1]
                    )
                if t > 0:
                    summands.append(
                        self.params.delta_del.get(source[t - 1], self.default)
                        + alpha[t - 1, v]
                    )
                if v > 0 and t > 0:
                    summands.append(
                        self.params.delta_sub.get(
                            (source[t - 1], target[v - 1]), self.default
                        )
                        + alpha[t - 1, v - 1]
                    )
                alpha[t, v] = special.logsumexp(summands)
        alpha[T, V] += self.params.delta_eos
        return alpha

    def backward_evaluate(
        self, source: Sequence[Any], target: Sequence[Any]
    ) -> numpy.ndarray:
        """Computes backward probabilities.

        Compute dynamic programming table (in log real) filled with backward
        log probabilities (the probabilities of the suffix, i.e.,
        p(source[t:], target[v:]). E.g., p("", "a") = p(ins(a)) * p(#).

        Args:
            source (Sequence[Any]): source string.
            target (Sequence[Any]): target string.

        Returns:
            numpy.ndarray: backward log probability alignments.
        """
        T = len(source)
        V = len(target)
        beta = numpy.full((T + 1, V + 1), LARGE_NEG_CONST)
        beta[T, V] = self.params.delta_eos
        for t in range(T, -1, -1):
            for v in range(V, -1, -1):
                summands = [beta[t, v]]
                if v < V:
                    summands.append(
                        self.params.delta_ins.get(target[v], self.default)
                        + beta[t, v + 1]
                    )
                if t < T:
                    summands.append(
                        self.params.delta_del.get(source[t], self.default)
                        + beta[t + 1, v]
                    )
                if v < V and t < T:
                    summands.append(
                        self.params.delta_sub.get(
                            (source[t], target[v]), self.default
                        )
                        + beta[t + 1, v + 1]
                    )
                beta[t, v] = special.logsumexp(summands)
        return beta

    def log_likelihood(
        self,
        sources: Iterable[Sequence[Any]],
        targets: Iterable[Sequence[Any]],
    ) -> float:
        """Computes log likelihood."""
        ll = numpy.mean(
            [
                self.forward_evaluate(source, target)[-1, -1]
                for source, target in zip(sources, targets)
            ]
        )
        return float(ll)

    def em(
        self,
        sources: Sequence[Any],
        targets: Sequence[Any],
        epochs: int = 10,
    ) -> None:
        """Update parameters using expectation-maximization.

        Args:
            sources (Sequence[Any]): source strings.
            targets (Sequence[Any]): target strings.
            epochs (int): number of EM epochs.
        """
        loglike = self.log_likelihood(sources, targets)
        gammas = ParamDict.from_params(self.params)
        for epoch in range(epochs):
            with tqdm.tqdm(zip(sources, targets), total=len(sources)) as pbar:
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix(loglike=loglike)
                for source, target in pbar:
                    self.e_step(source, target, gammas)  # Updates gammas.
                self.m_step(gammas)  # Updates gammas.
                self.params.update_params(gammas)  # Updates model parameters.
                loglike = self.log_likelihood(sources, targets)
        util.log_info(f"Final log-likelihood: {loglike:.4f}")

    def e_step(
        self, source: Sequence[Any], target: Sequence[Any], gammas: ParamDict
    ) -> None:
        """Accumulates soft counts.

        Args:
            source (Sequence[Any]): source string.
            target (Sequence[Any]): target string.
            gammas (ParamDict): unnormalized log weights.
        """
        alpha = self.forward_evaluate(source, target)
        beta = self.backward_evaluate(source, target)
        gammas.delta_eos = special.logsumexp([gammas.delta_eos, 0.0])
        T = len(source)
        V = len(target)
        for t in range(T + 1):
            for v in range(V + 1):
                rest = beta[t, v] - alpha[T, V]
                schar = source[t - 1]
                tchar = target[v - 1]
                stpair = schar, tchar
                if t > 0 and schar in gammas.delta_del:
                    gammas.delta_del[schar] = special.logsumexp(
                        [
                            gammas.delta_del[schar],
                            alpha[t - 1, v]
                            + self.params.delta_del[schar]
                            + rest,
                        ]
                    )
                if v > 0 and tchar in gammas.delta_ins:
                    gammas.delta_ins[tchar] = special.logsumexp(
                        [
                            gammas.delta_ins[tchar],
                            alpha[t, v - 1]
                            + self.params.delta_ins[tchar]
                            + rest,
                        ]
                    )
                if t > 0 and v > 0 and stpair in gammas.delta_sub:
                    gammas.delta_sub[stpair] = special.logsumexp(
                        [
                            gammas.delta_sub[stpair],
                            alpha[t - 1, v - 1]
                            + self.params.delta_sub[stpair]
                            + rest,
                        ]
                    )

    def m_step(self, gammas: ParamDict) -> None:
        """Normalizes weights and stores them."""
        denom = gammas.sum()
        gammas.delta_sub = {
            k: (v - denom) for k, v in gammas.delta_sub.items()
        }
        gammas.delta_del = {
            k: (v - denom) for k, v in gammas.delta_del.items()
        }
        gammas.delta_ins = {
            k: (v - denom) for k, v in gammas.delta_ins.items()
        }
        gammas.delta_eos -= denom
        assert numpy.isclose(0.0, gammas.sum()), gammas.sum()

    def action_sequence(
        self,
        source: Sequence,
        target: Sequence,
    ) -> Tuple[float, List]:
        """Computes optimal edit sequences using Viterbi edit distance.

        Viterbi edit distance \\propto max_{edits} p(target, edit | source).

        Args:
            source (Sequence[Any]): source string.
            target (Sequence[Any]): target string.

        Returns:
            Tuple[List, float]: sequence of edits that gives optimal score
                and optimal score.
        """
        alpha = self._viterbi_matrix(source, target)
        alignment = []
        # Computes an optimal alignment.
        ind_w, ind_c = len(source), len(target)
        while ind_w >= 0 and ind_c >= 0:
            if ind_w == 0 and ind_c == 0:
                return alignment[::-1], alpha[len(source), len(target)]
            if ind_w == 0:
                # Can only go left, i.e. via insertions.
                ind_c -= ind_c
                # Minus 1 is due to offset.
                alignment.append(actions.Ins(target[ind_c]))
            elif ind_c == 0:
                # Can only go up, i.e. via deletions.
                ind_w -= ind_w
                # minus 1 is due to offset
                alignment.append(actions.Del(source[ind_w]))
            else:
                # Picks the smallest cost actions.
                pind_w = ind_w - 1
                pind_c = ind_c - 1
                action_idx = numpy.argmax(
                    [
                        alpha[pind_w, pind_c],
                        alpha[ind_w, pind_c],
                        alpha[pind_w, ind_c],
                    ]
                )
                if action_idx == 0:
                    action = actions.Sub(source[pind_w], target[pind_c])
                    ind_w = pind_w
                    ind_c = pind_c
                elif action_idx == 1:
                    action = actions.Ins(target[pind_c])
                    ind_c = pind_c
                else:
                    action = actions.Del(source[pind_w])
                    ind_w = pind_w
                alignment.append(action)

    def _viterbi_matrix(
        self, source: Sequence, target: Sequence
    ) -> numpy.ndarray:
        """Computes Viterbi edit distance matrix.

        Viterbi edit distance \\propto max_{edits} p(target, edit | source).

        Args:
            source (Sequence[Any]): source string.
            target (Sequence[Any]): target string.

        Returns:
            numpy.ndarray: matrix of probability scores. Can index
                source-target score with len(source), len(target)
        """
        T = len(source)
        V = len(target)
        alpha = numpy.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.0
        for t in range(T + 1):
            for v in range(V + 1):
                alternatives = [alpha[t, v]]
                if v > 0:
                    alternatives.append(
                        self.params.delta_ins.get(target[v - 1], self.default)
                        + alpha[t, v - 1]
                    )
                if t > 0:
                    alternatives.append(
                        self.params.delta_del.get(source[t - 1], self.default)
                        + alpha[t - 1, v]
                    )
                if v > 0 and t > 0:
                    alternatives.append(
                        self.params.delta_sub.get(
                            (source[t - 1], target[v - 1]), self.default
                        )
                        + alpha[t - 1, v - 1]
                    )
                alpha[t, v] = max(alternatives)
        alpha[T, V] += self.params.delta_eos
        return alpha

    def action_sequence_cost(
        self, x: Sequence[Any], y: Sequence[Any], x_offset: int, y_offset: int
    ) -> float:
        source = x[x_offset:]
        target = y[y_offset:]
        # Get matrix of values for calculation.
        viterbi_matrix = self._viterbi_matrix(source=source, target=target)
        # Index location of source and target.
        return -viterbi_matrix[len(source), len(target)]

    def action_cost(self, action: actions.Edit) -> float:
        if isinstance(action, actions.Del):
            return -self.params.delta_del.get(action.old, self.default)
        if isinstance(action, actions.Ins):
            return -self.params.delta_ins.get(action.new, self.default)
        if isinstance(action, actions.Sub):
            return -self.params.delta_sub.get(
                (action.old, action.new), self.default
            )
        if isinstance(action, actions.End):
            return -self.params.delta_eos
        if isinstance(action, actions.Copy):
            return -self.params.delta_sub.get(
                (action.old, action.old), self.default
            )
        raise SEDActionError(f"Unknown action: {action}")
