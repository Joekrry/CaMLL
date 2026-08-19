CC       := cc
CFLAGS   := -std=c11 -Wall -Wextra -Wpedantic -O2 -g -I.
LDLIBS   := -lm
AR       := ar

BUILD    := build

SRCS := $(wildcard src/*.c) $(wildcard alloc/*.c)
OBJS := $(SRCS:%.c=$(BUILD)/%.o)

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(TEST_SRCS:tests/%.c=$(BUILD)/%)

LIB := $(BUILD)/libcamll.a

.PHONY: all lib test clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $(OBJS)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%: tests/%.c $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(OBJS) $(LDLIBS) -o $@

test: $(TEST_BINS)
	@fail=0; for t in $(TEST_BINS); do ./$$t || fail=1; done; exit $$fail

clean:
	rm -rf $(BUILD)
