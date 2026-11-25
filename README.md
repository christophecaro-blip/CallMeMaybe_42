*This activity has been created as part of the 42 curriculum by ccaro.*

# call me maybe — function calling with constrained decoding

## Description

**call me maybe** turns natural-language requests into structured function calls, using a small
language model (**Qwen3-0.6B**, about 600 million parameters).

Given the prompt `"What is the sum of 2 and 3?"` and a list of available functions, the program does
not answer `5`. It produces the call that would compute the answer:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Small models are unreliable at producing structured output: asked politely for JSON, Qwen3-0.6B
answers in prose, wraps the JSON in Markdown, invents functions, or computes the result itself.
This project does not rely on the prompt to get the format right. It uses **constrained decoding**:
at every generation step, the program removes from the model's choices every token that would break
the JSON or the function's schema. The model still decides *what* to say (which function, which
values); the program guarantees *how* it is written.

Main features:

- **100 % valid, schema-compliant JSON**, by construction: function names come only from the
  definitions file, parameter names are copied from it, and every value matches its declared type
  (`number`, `integer`, `string`, `boolean`).
- **Function and arguments chosen by the model**, never by keyword matching.
- **Robust error handling**: missing, empty, binary or malformed input files, unwritable output,
  a model that cannot be loaded, or Ctrl+C all end with a one-line message and a non-zero exit code,
  never a traceback.
- **Fast**: the 11 provided prompts take about 6 s on a GPU and about 70 s on a CPU (limit: 5 minutes).

## Instructions

### Requirements

- Python 3.13 or later (tested with 3.14, the version installed at school) and [uv](https://docs.astral.sh/uv/)
- About 1.5 GB of free disk space for the model, downloaded from Hugging Face on the first run
  (internet access needed once). A GPU is used automatically if available, but is not required.

### Installation

```bash
uv sync          # or: make install
```

This installs `numpy`, `pydantic`, the provided `llm_sdk` package (with its own dependencies,
`torch` and `transformers`) and `accelerate`, which the SDK needs on a GPU without declaring it.
The project's own code only imports `numpy`, `pydantic`, `json` and the SDK's public methods.

### Running

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` (folders are created if needed) |

Exit codes: `0` on success, `1` on a reported error (message on stderr), `130` if interrupted.

### Makefile

| Rule | Effect |
|---|---|
| `make install` | install the dependencies (`uv sync`) |
| `make run` | run the program with the default files |
| `make debug` | run it under `pdb` |
| `make clean` | remove `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `make lint` | `flake8 .` and `mypy .` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |

## Example usage

Default files:

```bash
make run
```

Custom files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Extract of the output for the provided prompts:

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {"a": 265.0, "b": 345.0}
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "([0-9]+)",
      "replacement": "NUMBERS"
    }
  }
]
```

Errors are reported on one line, without a traceback:

```text
$ uv run python -m src --input missing.json
Error: missing.json: [Errno 2] No such file or directory: 'missing.json'

$ uv run python -m src --input broken.json
Error: broken.json: prompt #2: prompt: Field required
```

## Algorithm explanation

### How a language model generates text

A language model works on **tokens**, pieces of text taken from a fixed vocabulary (151 669 tokens
for Qwen3). For a sequence of token ids, the model returns one **logit** (a raw score) per possible
next token. Generation repeats three steps: score every token, pick one (here, the highest score:
*greedy* decoding), append it to the sequence.

**Constrained decoding** acts between scoring and picking: tokens that would make the output invalid
are excluded, so the model can only choose among valid continuations. If every step keeps the output
valid, and generation only stops in a valid final state, the result is valid.

### Overview: forcing the structure, generating the values

Once a function is chosen, the whole shape of the JSON is known in advance. The program therefore
**writes the structure itself** and lets the model fill in only what is uncertain:

```text
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
 └─ forced ─┘└── model ───┘└──── forced ────┘└m┘└ forced┘└m┘
```

1. The prompt is built with Qwen3's chat template: a system message listing the functions (name,
   typed parameters, description), the user request, and an **empty thinking block**
   (`<think>\n\n</think>\n\n`), which disables Qwen3's step-by-step reasoning mode.
2. `{"name": "` is forced, then the model writes the **function name** under a choice constraint.
3. For each parameter, in the order of the definition, the key is forced (`, "b":`) and the model
   writes the **value** under the constraint of the declared type.
4. The closing braces are forced. The text is parsed with `json.loads`, validated with pydantic, and
   `number` values are converted to floats (the expected output writes `2.0`, not `2`).

A forced token costs **no model call**: its id is simply appended. About 60 % of each answer is
written this way.

### Token-level validity

The JSON rules are defined on **characters**, but the model produces **multi-character tokens**.
The vocabulary contains hundreds of tokens that mix a quote with JSON punctuation, such as `",`,
`"}` or `":`. Each value type is therefore a small **state machine** reading one character at a
time, and:

> a token is allowed if and only if the state machine accepts **all** of its characters, in order.

| Type | Constraint |
|---|---|
| function name | `ChoiceConstraint`: the text must be a prefix of a defined name, or a full name followed by its closing quote |
| `boolean` | the same choice constraint, over `true` and `false` |
| `number` | automaton for `-?(0\|[1-9][0-9]*)(\.[0-9]+)?` (exponents not supported); it may end only in an accepting state, so `-` or `1.` can never be closed |
| `integer` | the same automaton, with `.` refused |
| `string` | an optional space and the opening quote, then any character except raw control characters, then the closing quote; a backslash must start a valid JSON escape (`\\`, `\"`, `\n`…), so a regex `\d` must be written `\\d` |

Each value ends with a known marker: its closing quote for a string, the next structural character
(`,` or `}`) for a number or a boolean.

### Choosing the token: lazy search

Testing all 151 936 candidates at every step would be slow in Python. Instead, token ids are sorted
by decreasing logit (`np.argsort`) and tested one by one; the first accepted one is kept. The result
is identical to masking every invalid token with `-inf` and taking the maximum, but the search
usually stops after a few candidates.

When the best token overflows the value, for instance `"}}` while only the closing quote `"` is
valid, the token is cut to its longest prefix that completes the value: the model "voted" to close
the value, and that choice is kept.

### Mapping token ids to text

Qwen3 uses byte-level BPE: in `vocab.json`, each character of a token stands for one byte (a space
is shown as `Ġ`, a newline as `Ċ`). The program rebuilds the real text of every token by reversing
GPT-2's byte-to-character table and decoding the bytes as UTF-8. Ids that must never be generated
are excluded:

- the 26 special tokens (`<|im_start|>`, `<think>`…), listed in `tokenizer.json`, not in `vocab.json`;
- tokens that are only a fragment of a multi-byte UTF-8 character;
- 267 ids that have a logit but no token at all: the model's output layer has 151 936 entries
  (a multiple of 128, for GPU efficiency), more than the 151 669 real tokens.

### Termination

Every value has a hard token limit, and every constraint can return a `closing()` text that ends
it validly (`0` for an empty number, `""` for an unopened string, the closing quote otherwise).
If the limit is reached, or if no token is allowed at some step, the value is closed this way.
Generation therefore always ends, and always with valid JSON.

## Design decisions

- **Force the structure, generate only the values.** It guarantees the schema, removes most model
  calls, and keeps the model's choices where they matter.
- **Force keys only up to the colon.** The model writes the space and, for a string, the opening
  quote itself. Qwen3 writes them glued to the value (`" -3"`, `' "/home'`); forcing them separately
  put the model in an unnatural position, where it dropped minus signs and the leading `/` of paths.
- **Constraints are independent of the model.** They only answer "can this text extend the value?",
  so they are tested in milliseconds, without loading the model.
- **Greedy decoding.** Deterministic and reproducible, which suits extraction. Scoring whole
  sequences was measured and did not change the remaining failures (see *Challenges*).
- **A stuck value is closed, not dropped.** A prompt always gets an answer, at worst with a degraded
  value: answers are matched to prompts by position, so a missing answer would shift all the
  following ones. Real bugs (invalid generated JSON, unknown function) are still reported on stderr.
- **One error boundary.** Expected errors are project exceptions (`InputFileError`, `ModelError`,
  `DecodingError`) converted into a message and an exit code in a single place. Errors raised by the
  SDK, whose exceptions are not documented, are wrapped into `ModelError`.
- **pydantic for every class**: data models, configuration, constraints, vocabulary and engine. The
  SDK object is a private attribute of the engine, so it is neither validated nor copied.
- **Names are JSON-escaped** before being forced, so a quote or a backslash in a function or
  parameter name cannot break the output.

Project layout:

```text
src/
├── __main__.py        entry point: load inputs, run the pipeline, write the output, report errors
├── cli.py             command-line options
├── models.py          pydantic models of the input and output files
├── io_utils.py        reading and validating inputs, writing the output
├── errors.py          project exceptions
├── prompting.py       chat prompt (Qwen3 template, thinking disabled)
├── pipeline.py        builds one call: forced structure + generated values
├── llm/engine.py      the only module that talks to llm_sdk
├── llm/vocabulary.py  token id → real text
├── decoding/constraints.py  state machines for names and value types
└── decoding/decoder.py      constrained greedy decoding (lazy search, overflow cutting)
```

## Performance analysis

### Accuracy

Measured with the grading tool provided with the subject, which calls each function with the
generated arguments and compares its result with the expected one:

| Set | Correct calls |
|---|---|
| public set (the provided prompts) | **11 / 11 (100 %)** |
| private set (6 other functions, `integer` parameters, file paths, SQL, templates) | **10 / 11 (90.9 %)** |
| additional personal set (booleans, integers, a function without parameters, 3 mixed parameters) | 12 / 16 (75 %) |

The function name was right in every case. The remaining failures are about the *meaning* of a
value, never its format:

- a double quote inside a string (`Say "hello" to {name}`) becomes an apostrophe (`Say 'hello'`):
  writing an escaped quote `\"` is rare in the model's training data;
- a boolean from an artificial test function was always `false`, its description containing the
  word "false" itself.

### Speed

Measured on the 11 provided prompts, model already downloaded:

| | GPU (RTX 4050) | CPU |
|---|---|---|
| model loading | 2.2 s | 2.1 s |
| whole run | **6.3 s** | **69 s** |
| average time per prompt | 0.35 s | 6.1 s |
| one model call | 0.026 s | about 0.5 s |
| model calls per prompt | 10 to 12 | 10 to 12 |
| forced tokens per prompt (no model call) | 16 to 17 | 16 to 17 |

- On a CPU, more than 99 % of the time is spent inside the model. The decoder itself (sorting
  candidates, running the state machines) is negligible, which is why no further optimisation of it
  was made.
- The SDK has no key-value cache: every call re-reads the whole sequence, so a call costs more as the
  prompt grows (0.29 s for 56 tokens, 1.11 s for 441 tokens on a CPU). Forcing the structure is
  therefore the main speed gain: about 60 % of each answer costs no model call.
- At about 6 s per prompt on a CPU, the 5-minute limit allows around 45 prompts.

### Reliability

- Every output is valid JSON matching the definitions, by construction, then checked again by
  `json.loads` and pydantic.
- 16 hostile cases were run against the real program (missing, empty, binary or malformed files,
  wrong schema, unknown parameter type, empty list, empty, very long, Unicode or off-topic prompts,
  names containing quotes or backslashes, unwritable output, Ctrl+C): no traceback, a correct exit
  code each time.
- A prompt that cannot be related to any function still gets a call (the output format requires an
  existing function): the model picks the most plausible one.

## Challenges faced

- **Prompting alone does not work.** Asked for JSON, the model wrote prose, then "thought aloud" in
  its `<think>` mode, then produced Markdown-wrapped JSON with an invented function
  (`{"name": "sum", "parameters": {"value": 5}}`). The chat template improved understanding, and the
  empty thinking block removed the reasoning, but only constrained decoding fixed the format.
- **Tokens are not characters.** Tokens such as `",` or `"}}` close a value *and* add punctuation.
  Accepting only tokens whose characters all pass the automaton solved validity, but at first the
  model's favourite closing tokens were refused, and regular expressions ran until the token limit.
  Cutting an overflowing token to the prefix that closes the value raised the provided set from
  73 % to 91 %.
- **Forced text must be split where the model would split it.** Forcing `"b": ` (with its space)
  made the model drop minus signs, because it naturally writes `" -3"` with a space-and-minus token.
  The same issue, for the opening quote of strings, removed the leading `/` of
  `/home/user/data.json`. Forcing keys only up to the colon fixed both, and raised the grading
  tool's score from 19/22 to 21/22.
- **A measurement is only valid in the conditions where it was made.** The `"asterisks"` →
  `"asterisk"` failure was first measured, with sequence probabilities, as a semantic limit of the
  model (36 % for `asterisk` against 0.05 % for `*`). Those measurements were made with the
  forced opening quote; once the model wrote it itself, it chose `*`.
- **Prompt instructions can do more harm than good.** Two generic instructions ("write symbols as
  characters, not as their names", "copy text values exactly") lowered the provided set from 91 % to
  82 %: the model applied them where they did not belong (writing `NUMBERS` as the regex). They were
  dropped.
- **Environment.** The model and the virtual environment did not fit in the available home
  directory; they were moved to a larger disk. On a GPU, the SDK also needs `accelerate`, which it
  does not declare.

## Testing strategy

As the subject requires, test programs were written to verify the project but are not part of the
submission.

- **Unit tests (pytest, 172 tests, about 2 s).** They do not load the model:
  - every state machine: accepted and refused texts, tokens split in different ways, overflow
    tokens, length limits, `closing()`, escapes (a regex round trip `\d` → `"\\d"` → `\d`);
  - the decoder, with a fake engine returning scripted logits;
  - the pipeline, with a fake decoder (escaped names, stuck values still giving a valid call,
    invalid generated JSON reported as an error);
  - the vocabulary (byte-level decoding, UTF-8 fragments), input/output helpers, command line,
    and `main` (exit codes, messages without traceback).
- **Accuracy.** An evaluation script compares the output with hand-written expected answers. For
  regular expressions it compares the *result* of the substitution, since several regexes are
  correct. Each change was measured on the provided set and on a personal extended set (16 prompts,
  7 functions), and kept only if neither set regressed.
- **Grading tool.** The tool provided with the subject was run on its public and private sets.
- **Robustness.** A script runs the real program on 16 hostile cases and checks the exit code and the
  absence of a traceback.
- **Performance.** A script counts model calls and forced tokens and times each step, on GPU and on
  CPU.
- **Static checks.** `make lint` and `make lint-strict` (flake8, mypy `--strict`) pass.

## Resources

### References

- Stephane Robert Blog <https://www.stephane-robert.info/docs/developper/programmation/python/llm/>
- RFC 8259, *The JSON Data Interchange Format*: <https://www.rfc-editor.org/rfc/rfc8259>
- pydantic documentation: <https://docs.pydantic.dev/latest/>
- NumPy documentation: <https://numpy.org/doc/stable/>
- uv documentation: <https://docs.astral.sh/uv/>

### Use of AI

An AI assistant (Claude, by Anthropic) was used throughout the project, as a tutor and a reviewer:

- **Learning**: explanations of tokenization, logits, softmax, constrained decoding and state
  machines, and a step-by-step roadmap of the project.
- **Review**: a full audit of the code, which found two crashes (a non-UTF-8 input file, a model that
  cannot be loaded) and two non-compliant classes; the assistant wrote these fixes and their tests.
- **Documentation**: a first version of this README, written from the project's notes and reviewed.
