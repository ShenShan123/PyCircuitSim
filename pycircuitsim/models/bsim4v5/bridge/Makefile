# Makefile for BSIM4 Standalone Wrapper
# Author: PyCircuitSim Team

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -I.. -I.
LDFLAGS = -shared -lm

# Source files
WRAPPER_SRC = bsim4_wrapper.c
IV_CORE_SRC = bsim4_iv_core.c

# Output library
LIB = libbsim4.so

# Object files
WRAPPER_OBJ = $(WRAPPER_SRC:.c=.o)
IV_CORE_OBJ = $(IV_CORE_SRC:.c=.o)

# Default target
all: $(LIB)

# Link the shared library
$(LIB): $(IV_CORE_OBJ) $(WRAPPER_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	rm -f $(LIB) $(IV_CORE_OBJ) $(WRAPPER_OBJ)

# Test target
test: $(LIB)
	python3 test_bsim4.py

.PHONY: all clean test
