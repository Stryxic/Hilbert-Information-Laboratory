#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "hilbert_native.h"

/* --------------------------------------------------------------------- */
/* Helpers: Python sequence -> C double[]                                */
/* --------------------------------------------------------------------- */

static int pyseq_to_double_array(PyObject *seq, double **out, int *n) {
    if (!PySequence_Check(seq)) return 0;

    Py_ssize_t len = PySequence_Size(seq);
    if (len < 0) return 0;
    if (len == 0) {
        *out = NULL;
        *n = 0;
        return 1;
    }

    double *buf = (double *) malloc(sizeof(double) * (size_t)len);
    if (!buf) return 0;

    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);  /* new ref */
        if (!item) {
            free(buf);
            return 0;
        }
        double v = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            free(buf);
            return 0;
        }
        buf[i] = v;
    }

    *out = buf;
    *n = (int) len;
    return 1;
}

/* --------------------------------------------------------------------- */
/* Lifecycle                                                            */
/* --------------------------------------------------------------------- */

static PyObject *py_init(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    if (!hilbert_init()) {
        PyErr_SetString(PyExc_RuntimeError, "hilbert_init failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_shutdown(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    hilbert_shutdown();
    Py_RETURN_NONE;
}

static PyObject *py_version(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return PyFloat_FromDouble(hilbert_version());
}

/* --------------------------------------------------------------------- */
/* Spectral primitives                                                  */
/* --------------------------------------------------------------------- */

static PyObject *py_spectral_entropy(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;

    double *v = NULL;
    int n = 0;
    if (!pyseq_to_double_array(seq, &v, &n)) {
        PyErr_SetString(PyExc_TypeError, "Expected numeric sequence");
        return NULL;
    }

    double e = hilbert_spectral_entropy(v, n);
    free(v);
    return PyFloat_FromDouble(e);
}

static PyObject *py_spectral_coherence(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj = NULL;
    PyObject *b_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;

    double *a = NULL; int na = 0;
    double *b = NULL; int nb = 0;

    if (!pyseq_to_double_array(a_obj, &a, &na) ||
        !pyseq_to_double_array(b_obj, &b, &nb)) {
        if (a) free(a);
        if (b) free(b);
        PyErr_SetString(PyExc_TypeError, "Expected numeric sequences");
        return NULL;
    }

    int n = (na < nb) ? na : nb;
    double c = hilbert_coherence_score(a, n, 1);  // updated to match C API

    free(a);
    free(b);
    return PyFloat_FromDouble(c);
}

static PyObject *py_information_stability(PyObject *self, PyObject *args) {
    (void)self;
    double entropy, coherence;
    if (!PyArg_ParseTuple(args, "dd", &entropy, &coherence)) return NULL;
    double s = hilbert_information_stability(entropy, coherence);
    return PyFloat_FromDouble(s);
}

/* --------------------------------------------------------------------- */
/* Graph export                                                         */
/* --------------------------------------------------------------------- */

static PyObject *py_graph_export_edges(PyObject *self, PyObject *args) {
    (void)self;
    const char *path = NULL;
    PyObject *edges_obj = NULL;

    if (!PyArg_ParseTuple(args, "sO", &path, &edges_obj)) return NULL;
    if (!PySequence_Check(edges_obj)) {
        PyErr_SetString(PyExc_TypeError, "edges must be a sequence");
        return NULL;
    }

    Py_ssize_t n = PySequence_Size(edges_obj);
    if (n <= 0) Py_RETURN_NONE;

    char **src = (char **) malloc(sizeof(char *) * (size_t)n);
    char **tgt = (char **) malloc(sizeof(char *) * (size_t)n);
    double *w  = (double *) malloc(sizeof(double) * (size_t)n);

    if (!src || !tgt || !w) {
        free(src); free(tgt); free(w);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(edges_obj, i);
        if (!item) {
            PyErr_SetString(PyExc_TypeError, "Invalid edge item");
            goto fail;
        }

        PyObject *s_obj = PySequence_GetItem(item, 0);
        PyObject *t_obj = PySequence_GetItem(item, 1);
        PyObject *w_obj = PySequence_GetItem(item, 2);

        const char *s_utf8 = PyUnicode_AsUTF8(s_obj);
        const char *t_utf8 = PyUnicode_AsUTF8(t_obj);
        double ww = PyFloat_AsDouble(w_obj);

        Py_XDECREF(s_obj); Py_XDECREF(t_obj); Py_XDECREF(w_obj); Py_XDECREF(item);
        if (PyErr_Occurred()) goto fail;

        src[i] = (char *)(s_utf8 ? s_utf8 : "");
        tgt[i] = (char *)(t_utf8 ? t_utf8 : "");
        w[i]   = ww;
    }

    if (!hilbert_graph_export_edges(path,
                                    (const char **)src,
                                    (const char **)tgt,
                                    w,
                                    (int)n)) {
        PyErr_SetString(PyExc_RuntimeError, "hilbert_graph_export_edges failed");
        goto fail;
    }

    free(src); free(tgt); free(w);
    Py_RETURN_NONE;

fail:
    free(src); free(tgt); free(w);
    return NULL;
}

/* --------------------------------------------------------------------- */
/* Store helpers                                                        */
/* --------------------------------------------------------------------- */

static PyObject *py_set_output_dir(PyObject *self, PyObject *args) {
    (void)self;
    const char *path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    hilbert_set_output_dir(path);
    Py_RETURN_NONE;
}

static PyObject *py_read_store(PyObject *self, PyObject *args) {
    (void)self;
    const char *kind = NULL;
    if (!PyArg_ParseTuple(args, "s", &kind)) return NULL;

    char *json = hilbert_read_store(kind);
    if (!json) Py_RETURN_NONE;

    PyObject *s = PyUnicode_FromString(json);
    hilbert_free(json);
    return s;
}

/* --------------------------------------------------------------------- */
/* Simulation API Wrappers                                              */
/* --------------------------------------------------------------------- */

static PyObject *py_sim_init(PyObject *self, PyObject *args) {
    (void)self;
    int n_agents = 0;
    if (!PyArg_ParseTuple(args, "i", &n_agents)) return NULL;
    int ok = hilbert_sim_init(n_agents);
    return PyBool_FromLong(ok);
}

static PyObject *py_sim_step(PyObject *self, PyObject *args) {
    (void)self;
    double dt;
    if (!PyArg_ParseTuple(args, "d", &dt)) return NULL;
    hilbert_sim_step(dt);
    Py_RETURN_NONE;
}

static PyObject *py_sim_inject(PyObject *self, PyObject *args) {
    (void)self;
    double x, y, r;
    int regime;
    if (!PyArg_ParseTuple(args, "dddi", &x, &y, &r, &regime)) return NULL;
    hilbert_sim_inject(x, y, r, regime);
    Py_RETURN_NONE;
}

static PyObject *py_sim_metrics(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    double mean, pol, ent;
    hilbert_sim_metrics(&mean, &pol, &ent);
    return Py_BuildValue("{s:d,s:d,s:d}", "mean", mean, "polarization", pol, "entropy", ent);
}

static PyObject *py_sim_export_state(PyObject *self, PyObject *args) {
    (void)self;
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    hilbert_sim_export_state(path);
    Py_RETURN_NONE;
}

/* --------------------------------------------------------------------- */
/* Module definition                                                    */
/* --------------------------------------------------------------------- */

static PyMethodDef HilbertMethods[] = {
    {"init",                py_init,                METH_NOARGS,  "Initialize hilbert_native"},
    {"shutdown",            py_shutdown,            METH_NOARGS,  "Shutdown hilbert_native"},
    {"version",             py_version,             METH_NOARGS,  "Get hilbert_native version"},
    {"spectral_entropy",    py_spectral_entropy,    METH_VARARGS, "Compute spectral entropy"},
    {"spectral_coherence",  py_spectral_coherence,  METH_VARARGS, "Compute spectral coherence"},
    {"information_stability", py_information_stability, METH_VARARGS, "Compute information stability"},
    {"graph_export_edges",  py_graph_export_edges,  METH_VARARGS, "Export edges to CSV"},
    {"set_output_dir",      py_set_output_dir,      METH_VARARGS, "Set base output directory"},
    {"read_store",          py_read_store,          METH_VARARGS, "Read JSON store"},
    {"sim_init",            py_sim_init,            METH_VARARGS, "Initialize agent simulation"},
    {"sim_step",            py_sim_step,            METH_VARARGS, "Advance simulation step"},
    {"sim_inject",          py_sim_inject,          METH_VARARGS, "Inject information/misinformation/disinformation"},
    {"sim_metrics",         py_sim_metrics,         METH_NOARGS,  "Get simulation metrics"},
    {"sim_export_state",    py_sim_export_state,    METH_VARARGS, "Export simulation state to JSON"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hilbertmodule = {
    PyModuleDef_HEAD_INIT,
    "hilbert_native",
    "Hilbert native core and simulation bindings",
    -1,
    HilbertMethods
};

PyMODINIT_FUNC PyInit_hilbert_native(void) {
    (void)hilbert_init(); // safe default init
    return PyModule_Create(&hilbertmodule);
}
