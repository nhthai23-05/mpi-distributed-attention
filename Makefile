CC      = mpicc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm
TARGET  = hybrid_attention

SRCS = src/main.c \
       src/tensor.c \
       src/data_gen.c \
       src/profiler.c \
       src/attention.c \
       src/head_parallel.c \
       src/tensor_parallel.c \
       src/hybrid.c

OBJS = $(SRCS:.c=.o)

all: $(TARGET) mpi-prime

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Cluster smoke-test program (Part 4 of the setup instruction).
# Third-party file (Burkardt) — built standalone with plain flags so its
# harmless unused-variable warnings don't clutter the project build.
mpi-prime: mpi-prime.c
	$(CC) -O2 -std=c11 -o $@ $< -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) mpi-prime

.PHONY: all clean
