# backend/native/Makefile
#
# Build the Hilbert native engine as a Python extension module.
# Outputs something like: hilbert_native.cpython-310-x86_64-linux-gnu.so

PYTHON ?= python
PYTHON_CONFIG ?= $(PYTHON)-config

# Detect extension suffix for this Python (e.g. .cpython-310-x86_64-linux-gnu.so)
EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix 2>/dev/null || $(PYTHON) - <<EOF
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
EOF
)

CC      ?= gcc
CFLAGS  ?= -O3 -fPIC -std=c11 -Wall -Wextra -I.
LDFLAGS ?= -shared

# Python include and library flags
PY_CFLAGS  := $(shell $(PYTHON_CONFIG) --cflags)
PY_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

TARGET = hilbert_native$(EXT_SUFFIX)

CORE_SRCS = \
    hilbert_math.c \
    hilbert_matrix.c \
    hilbert_random.c \
    hilbert_stats.c \
    hilbert_entropy.c

ENGINE_SRCS = \
    hilbert_simulation.c \
    hilbert_native.c \
    hilbert_pybind.c

SRCS = $(CORE_SRCS) $(ENGINE_SRCS)
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(PY_LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(PY_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
