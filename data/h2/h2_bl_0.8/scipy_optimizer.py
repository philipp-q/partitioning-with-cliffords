import numpy, copy, scipy, typing, numbers

from tequila import BitString, BitNumbering, BitStringLSB
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila.circuit.compiler import change_basis
from tequila.utils import to_float

import tequila as tq
from tequila.objective import Objective
from tequila.optimizers.optimizer_scipy import OptimizerSciPy, SciPyResults
from tequila.objective.objective import assign_variable, Variable, format_variable_dictionary, format_variable_list
from tequila.circuit.noise import NoiseModel
#from tequila.optimizers._containers import _EvalContainer, _GradContainer, _HessContainer, _QngContainer
from vqe_utils import *

class _EvalContainer:
    """
    Overwrite the call function

    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.
    Attributes
    ---------
    objective:
        the objective to evaluate.
    param_keys:
        the dictionary mapping parameter keys to positions in a numpy array.
    samples:
        the number of samples to evaluate objective with.
    save_history:
        whether or not to save, in a history, information about each time __call__ occurs.
    print_level
        dictates the verbosity of printing during call.
    N:
        the length of param_keys.
    history:
        if save_history, a list of energies received from every __call__
    history_angles:
        if save_history, a list of angles sent to __call__.
    """

    def __init__(self, Hamiltonian, unitary, param_keys, Ham_derivatives= None, Eval=None, passive_angles=None, samples=1024, save_history=True,
                 print_level: int = 3):
        self.Hamiltonian = Hamiltonian
        self.unitary = unitary
        self.samples = samples
        self.param_keys = param_keys
        self.N = len(param_keys)
        self.save_history = save_history
        self.print_level = print_level
        self.passive_angles = passive_angles
        self.Eval = Eval
        self.infostring = None
        self.Ham_derivatives = Ham_derivatives
        if save_history:
            self.history = []
            self.history_angles = []

    def __call__(self, p, *args, **kwargs):
        """
        call a wrapped objective.
        Parameters
        ----------
        p: numpy array:
            Parameters with which to call the objective.
        args
        kwargs
        Returns
        -------
        numpy.array:
            value of self.objective with p translated into variables, as a numpy array.
        """

        angles = {}#dict((self.param_keys[i], p[i]) for i in range(self.N))
        for i in range(self.N):
            if self.param_keys[i] in self.unitary.extract_variables():
                angles[self.param_keys[i]] = p[i]
            else:
                angles[self.param_keys[i]] = complex(p[i])

        if self.passive_angles is not None:
            angles = {**angles, **self.passive_angles}
        vars = format_variable_dictionary(angles)

        Hamiltonian = self.Hamiltonian(vars)
        #print(Hamiltonian)
        #print(self.unitary)
        #print(vars)
        Expval = tq.ExpectationValue(H=Hamiltonian, U=self.unitary)
        #print(Expval)
        E = tq.simulate(Expval, vars, backend='qulacs', samples=self.samples)

        self.infostring = "{:15} : {} expectationvalues\n".format("Objective", Expval.count_expectationvalues())

        if self.print_level > 2:
            print("E={:+2.8f}".format(E), " angles=", angles, " samples=", self.samples)
        elif self.print_level > 1:
            print("E={:+2.8f}".format(E))
        if self.save_history:
            self.history.append(E)
            self.history_angles.append(angles)
        return complex(E)  # jax types confuses optimizers

class _GradContainer(_EvalContainer):
    """
    Overwrite the call function

    Container Class to access scipy and keep the optimization history.
    This class is used by the SciPy optimizer and should not be used elsewhere.
    see _EvalContainer for details.
    """

    def __call__(self, p, *args, **kwargs):
        """
        call the wrapped qng.
        Parameters
        ----------
        p: numpy array:
            Parameters with which to call gradient
        args
        kwargs
        Returns
        -------
        numpy.array:
            value of self.objective with p translated into variables, as a numpy array.
        """
        Ham_derivatives = self.Ham_derivatives
        Hamiltonian = self.Hamiltonian
        unitary = self.unitary

        dE_vec = numpy.zeros(self.N)
        memory = dict()
        #variables = dict((self.param_keys[i], p[i]) for i in range(len(self.param_keys)))
        variables = {}#dict((self.param_keys[i], p[i]) for i in range(self.N))
        for i in range(len(self.param_keys)):
            if self.param_keys[i] in self.unitary.extract_variables():
                variables[self.param_keys[i]] = p[i]
            else:
                variables[self.param_keys[i]] = complex(p[i])

        if self.passive_angles is not None:
            variables = {**variables, **self.passive_angles}
        vars = format_variable_dictionary(variables)
        expvals = 0
        for i in range(self.N):
            derivative = 0.0
            if self.param_keys[i] in list(unitary.extract_variables()):
                Ham = Hamiltonian(vars)
                Expval = tq.ExpectationValue(H=Ham, U=unitary)
                temp_derivative = tq.compile(objective = tq.grad(objective = Expval, variable = self.param_keys[i]),backend='qulacs')
                expvals += temp_derivative.count_expectationvalues()
                derivative += temp_derivative

            if self.param_keys[i] in list(Ham_derivatives.keys()):
                #print(self.param_keys[i])
                Ham = Ham_derivatives[self.param_keys[i]]
                Ham = convert_PQH_to_tq_QH(Ham)
                H = Ham(vars)
                #print(H)
                #raise Exception("testing")
                Expval = tq.ExpectationValue(H=H, U=unitary)
                expvals += Expval.count_expectationvalues()
                derivative += tq.simulate(Expval, vars, backend='qulacs', samples=self.samples)
                #print(derivative)
                #print(type(H))

            if isinstance(derivative, float) or isinstance(derivative, numpy.complex64) :
                dE_vec[i] = derivative
            else:
                dE_vec[i] = derivative(variables=variables, samples=self.samples)

            memory[self.param_keys[i]] = dE_vec[i]
        self.infostring = "{:15} : {} expectationvalues\n".format("gradient", expvals)
        self.history.append(memory)
        return numpy.asarray(dE_vec, dtype=numpy.complex64)

class optimize_scipy(OptimizerSciPy):
    """
    overwrite the expectation and gradient container objects
    """

    def initialize_variables(self, all_variables, initial_values, variables):
        """
        Convenience function to format the variables of some objective recieved in calls to optimzers.
        Parameters
        ----------
        objective: Objective:
            the objective being optimized.
        initial_values: dict or string:
            initial values for the variables of objective, as a dictionary.
            if string: can be `zero` or `random`
            if callable: custom function that initializes when keys are passed
            if None: random initialization between 0 and 2pi (not recommended)
        variables: list:
            the variables being optimized over.
        Returns
        -------
        tuple:
            active_angles, a dict of those variables being optimized.
            passive_angles, a dict of those variables NOT being optimized.
            variables: formatted list of the variables being optimized.
        """
        # bring into right format
        variables = format_variable_list(variables)
        initial_values = format_variable_dictionary(initial_values)
        all_variables = all_variables

        if variables is None:
            variables = all_variables
        if initial_values is None:
            initial_values = {k: numpy.random.uniform(0, 2 * numpy.pi) for k in all_variables}
        elif hasattr(initial_values, "lower"):
            if initial_values.lower() == "zero":
                initial_values = {k:0.0 for k in all_variables}
            elif initial_values.lower() == "random":
                initial_values = {k: numpy.random.uniform(0, 2 * numpy.pi) for k in all_variables}
            else:
                raise TequilaOptimizerException("unknown initialization instruction: {}".format(initial_values))
        elif callable(initial_values):
            initial_values = {k: initial_values(k) for k in all_variables}
        elif isinstance(initial_values, numbers.Number):
            initial_values = {k: initial_values for k in all_variables}
        else:
            # autocomplete initial values, warn if you did
            detected = False
            for k in all_variables:
                if k not in initial_values:
                    initial_values[k] = 0.0
                    detected = True
            if detected and not self.silent:
                warnings.warn("initial_variables given but not complete: Autocompleted with zeroes", TequilaWarning)

        active_angles = {}
        for v in variables:
            active_angles[v] = initial_values[v]

        passive_angles = {}
        for k, v in initial_values.items():
            if k not in active_angles.keys():
                passive_angles[k] = v
        return active_angles, passive_angles, variables

    def __call__(self, Hamiltonian, unitary,
                 variables: typing.List[Variable] = None,
                 initial_values: typing.Dict[Variable, numbers.Real] = None,
                 gradient: typing.Dict[Variable, Objective] = None,
                 hessian: typing.Dict[typing.Tuple[Variable, Variable], Objective] = None,
                 reset_history: bool = True,
                 *args,
                 **kwargs) -> SciPyResults:

        """
        Perform optimization using scipy optimizers.

        Parameters
        ----------
        objective: Objective:
            the objective to optimize.
        variables: list, optional:
            the variables of objective to optimize. If None: optimize all.
        initial_values: dict, optional:
            a starting point from which to begin optimization. Will be generated if None.
        gradient: optional:
            Information or object used to calculate the gradient of objective. Defaults to None: get analytically.
        hessian: optional:
            Information or object used to calculate the hessian of objective. Defaults to None: get analytically.
        reset_history: bool: Default = True:
            whether or not to reset all history before optimizing.
        args
        kwargs

        Returns
        -------
        ScipyReturnType:
            the results of optimization.
        """
        H = convert_PQH_to_tq_QH(Hamiltonian)
        Ham_variables, Ham_derivatives = H._construct_derivatives()
        #print("hamvars",Ham_variables)
        all_variables = copy.deepcopy(Ham_variables)
        #print(all_variables)
        for var in unitary.extract_variables():
            all_variables.append(var)

        #print(all_variables)
        infostring = "{:15} : {}\n".format("Method", self.method)
        #infostring += "{:15} : {} expectationvalues\n".format("Objective", objective.count_expectationvalues())

        if self.save_history and reset_history:
            self.reset_history()

        active_angles, passive_angles, variables = self.initialize_variables(all_variables, initial_values, variables)
        #print(active_angles, passive_angles, variables)
        # Transform the initial value directory into (ordered) arrays
        param_keys, param_values = zip(*active_angles.items())
        param_values = numpy.array(param_values)
        # process and initialize scipy bounds
        bounds = None
        if self.method_bounds is not None:
            bounds = {k: None for k in active_angles}
            for k, v in self.method_bounds.items():
                if k in bounds:
                    bounds[k] = v
            infostring += "{:15} : {}\n".format("bounds", self.method_bounds)
            names, bounds = zip(*bounds.items())
            assert (names == param_keys)  # make sure the bounds are not shuffled

        #print(param_keys, param_values)

        # do the compilation here to avoid costly recompilation during the optimization
        #compiled_objective = self.compile_objective(objective=objective, *args, **kwargs)
        E = _EvalContainer(Hamiltonian = H,
                           unitary = unitary,
                           Eval=None,
                           param_keys=param_keys,
                           samples=self.samples,
                           passive_angles=passive_angles,
                           save_history=self.save_history,
                           print_level=self.print_level)

        E.print_level = 0
        (E(param_values))
        E.print_level = self.print_level
        infostring += E.infostring

        if gradient is not None:
            infostring += "{:15} : {}\n".format("grad instr", gradient)
        if hessian is not None:
            infostring += "{:15} : {}\n".format("hess_instr", hessian)

        compile_gradient = self.method in (self.gradient_based_methods + self.hessian_based_methods)
        compile_hessian = self.method in self.hessian_based_methods

        dE = None
        ddE = None
        # detect if numerical gradients shall be used
        # switch off compiling if so
        if isinstance(gradient, str):
            if gradient.lower() == 'qng':
                compile_gradient = False
                if compile_hessian:
                    raise TequilaException('Sorry, QNG and hessian not yet tested together.')

                combos = get_qng_combos(objective, initial_values=initial_values, backend=self.backend,
                                        samples=self.samples, noise=self.noise)
                dE = _QngContainer(combos=combos, param_keys=param_keys, passive_angles=passive_angles)
                infostring += "{:15} : QNG {}\n".format("gradient", dE)
            else:
                dE = gradient
                compile_gradient = False
                if compile_hessian:
                    compile_hessian = False
                    if hessian is None:
                        hessian = gradient
                infostring += "{:15} : scipy numerical {}\n".format("gradient", dE)
                infostring += "{:15} : scipy numerical {}\n".format("hessian", ddE)

        if isinstance(gradient,dict):
            if gradient['method'] == 'qng':
                func = gradient['function']
                compile_gradient = False
                if compile_hessian:
                    raise TequilaException('Sorry, QNG and hessian not yet tested together.')

                combos = get_qng_combos(objective,func=func, initial_values=initial_values, backend=self.backend,
                                        samples=self.samples, noise=self.noise)
                dE = _QngContainer(combos=combos, param_keys=param_keys, passive_angles=passive_angles)
                infostring += "{:15} : QNG {}\n".format("gradient", dE)

        if isinstance(hessian, str):
            ddE = hessian
            compile_hessian = False

        if compile_gradient:
            dE =_GradContainer(Ham_derivatives = Ham_derivatives,
                               unitary = unitary,
                               Hamiltonian = H,
                               Eval= E,
                               param_keys=param_keys,
                               samples=self.samples,
                               passive_angles=passive_angles,
                               save_history=self.save_history,
                               print_level=self.print_level)

            dE.print_level = 0
            (dE(param_values))
            dE.print_level = self.print_level
            infostring += dE.infostring

        if self.print_level > 0:
            print(self)
            print(infostring)
            print("{:15} : {}\n".format("active variables", len(active_angles)))

        Es = []

        optimizer_instance = self
        class SciPyCallback:
            energies = []
            gradients = []
            hessians = []
            angles = []
            real_iterations = 0

            def __call__(self, *args, **kwargs):
                self.energies.append(E.history[-1])
                self.angles.append(E.history_angles[-1])
                if dE is not None and not isinstance(dE, str):
                    self.gradients.append(dE.history[-1])
                if ddE is not None and not isinstance(ddE, str):
                    self.hessians.append(ddE.history[-1])
                self.real_iterations += 1
                if 'callback' in optimizer_instance.kwargs:
                    optimizer_instance.kwargs['callback'](E.history_angles[-1])

        callback = SciPyCallback()
        res = scipy.optimize.minimize(E, x0=param_values, jac=dE, hess=ddE,
                                      args=(Es,),
                                      method=self.method, tol=self.tol,
                                      bounds=bounds,
                                      constraints=self.method_constraints,
                                      options=self.method_options,
                                      callback=callback)

        # failsafe since callback is not implemented everywhere
        if callback.real_iterations == 0:
            real_iterations = range(len(E.history))

        if self.save_history:
            self.history.energies = callback.energies
            self.history.energy_evaluations = E.history
            self.history.angles = callback.angles
            self.history.angles_evaluations = E.history_angles
            self.history.gradients = callback.gradients
            self.history.hessians = callback.hessians
            if dE is not None and not isinstance(dE, str):
                self.history.gradients_evaluations = dE.history
            if ddE is not None and not isinstance(ddE, str):
                self.history.hessians_evaluations = ddE.history

            # some methods like "cobyla" do not support callback functions
            if len(self.history.energies) == 0:
                self.history.energies = E.history
                self.history.angles = E.history_angles

        # some scipy methods always give back the last value and not the minimum (e.g. cobyla)
        ea = sorted(zip(E.history, E.history_angles), key=lambda x: x[0])
        E_final = ea[0][0]
        angles_final = ea[0][1] #dict((param_keys[i], res.x[i]) for i in range(len(param_keys)))
        angles_final = {**angles_final, **passive_angles}

        return SciPyResults(energy=E_final, history=self.history, variables=format_variable_dictionary(angles_final), scipy_result=res)

def minimize(Hamiltonian, unitary,
             gradient: typing.Union[str, typing.Dict[Variable, Objective]] = None,
             hessian: typing.Union[str, typing.Dict[typing.Tuple[Variable, Variable], Objective]] = None,
             initial_values: typing.Dict[typing.Hashable, numbers.Real] = None,
             variables: typing.List[typing.Hashable] = None,
             samples: int = None,
             maxiter: int = 100,
             backend: str = None,
             backend_options: dict = None,
             noise: NoiseModel = None,
             device: str = None,
             method: str = "BFGS",
             tol: float = 1.e-3,
             method_options: dict = None,
             method_bounds: typing.Dict[typing.Hashable, numbers.Real] = None,
             method_constraints=None,
             silent: bool = False,
             save_history: bool = True,
             *args,
             **kwargs) -> SciPyResults:
    """
    calls the local optimize_scipy scipy funtion instead and pass down the objective construction
    down

    Parameters
    ----------
    objective: Objective :
        The tequila objective to optimize
    gradient: typing.Union[str, typing.Dict[Variable, Objective], None] : Default value = None):
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary of variables and tequila objective to define own gradient,
        None for automatic construction (default)
        Other options include 'qng' to use the quantum natural gradient.
    hessian: typing.Union[str, typing.Dict[Variable, Objective], None], optional:
        '2-point', 'cs' or '3-point' for numerical gradient evaluation (does not work in combination with all optimizers),
        dictionary (keys:tuple of variables, values:tequila objective) to define own gradient,
        None for automatic construction (default)
    initial_values: typing.Dict[typing.Hashable, numbers.Real], optional:
        Initial values as dictionary of Hashable types (variable keys) and floating point numbers. If given None they will all be set to zero
    variables: typing.List[typing.Hashable], optional:
         List of Variables to optimize
    samples: int, optional:
         samples/shots to take in every run of the quantum circuits (None activates full wavefunction simulation)
    maxiter: int : (Default value = 100):
         max iters to use.
    backend: str, optional:
         Simulator backend, will be automatically chosen if set to None
    backend_options: dict, optional:
         Additional options for the backend
         Will be unpacked and passed to the compiled objective in every call
    noise: NoiseModel, optional:
         a NoiseModel to apply to all expectation values in the objective.
    method: str : (Default = "BFGS"):
         Optimization method (see scipy documentation, or 'available methods')
    tol: float : (Default = 1.e-3):
         Convergence tolerance for optimization (see scipy documentation)
    method_options: dict, optional:
         Dictionary of options
         (see scipy documentation)
    method_bounds: typing.Dict[typing.Hashable, typing.Tuple[float, float]], optional:
        bounds for the variables (see scipy documentation)
    method_constraints: optional:
         (see scipy documentation
    silent: bool :
         No printout if True
    save_history: bool:
        Save the history throughout the optimization

    Returns
    -------
    SciPyReturnType:
        the results of optimization
    """

    if isinstance(gradient, dict) or hasattr(gradient, "items"):
        if all([isinstance(x, Objective) for x in gradient.values()]):
            gradient = format_variable_dictionary(gradient)
    if isinstance(hessian, dict) or hasattr(hessian, "items"):
        if all([isinstance(x, Objective) for x in hessian.values()]):
            hessian = {(assign_variable(k[0]), assign_variable([k[1]])): v for k, v in hessian.items()}
    method_bounds = format_variable_dictionary(method_bounds)

    # set defaults

    optimizer = optimize_scipy(save_history=save_history,
                               maxiter=maxiter,
                               method=method,
                               method_options=method_options,
                               method_bounds=method_bounds,
                               method_constraints=method_constraints,
                               silent=silent,
                               backend=backend,
                               backend_options=backend_options,
                               device=device,
                               samples=samples,
                               noise_model=noise,
                               tol=tol,
                               *args,
                               **kwargs)
    if initial_values is not None:
        initial_values = {assign_variable(k): v for k, v in initial_values.items()}
    return optimizer(Hamiltonian, unitary,
                     gradient=gradient,
                     hessian=hessian,
                     initial_values=initial_values,
                     variables=variables, *args, **kwargs)
