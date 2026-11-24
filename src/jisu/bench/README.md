# jisu 벤치마크 (`jisu.bench`)

## 기억 검색 벤치마크 (`jisu.bench.memory`)

LLM 없이 기억 저장소를 직접 호출한다. 한국어 사실을 저장한 뒤 질의로 찾아서, **얼마나 잘 찾는지(정확도)**와 **얼마나 빠른지(지연)**를 잰다.

### 실행

```bash
# 기본 실행: json, gbrain, gbrain-native 세 저장소에 같은 사실 30개를 넣고 질문 41개로 찾아 본다.
# 질문마다 상위 5개(k=5)를 보고, 속도는 질문당 3번(repeat=3) 재서 p50·p95를 낸다. 결과 파일을 저장한다.
python -m jisu.bench.memory

# 저장소 두 개만 비교한다. 상위 3개만 정답으로 인정하고(k=3, 더 엄격), 속도 표본을 질문당 5번으로 늘린다.
python -m jisu.bench.memory --subjects json gbrain --k 3 --repeat 5

# 기본 실행과 같지만, 끝에 틀린 질문 목록(기대한 사실 vs 실제로 나온 사실)을 함께 출력한다. 왜 틀렸는지 볼 때 쓴다.
python -m jisu.bench.memory --detail

# 내가 만든 데이터셋으로 시험한다. 결과 파일은 남기지 않는다(시험 삼아 돌려 볼 때).
python -m jisu.bench.memory --dataset my.toml --no-save

# 지금까지 저장된 결과 파일 목록을 최신순으로 보여 준다(측정은 하지 않음).
python -m jisu.bench.memory --list

# 저장된 결과 파일 하나를 표로 다시 출력한다(측정은 하지 않음).
python -m jisu.bench.memory --show <파일>
```

- 실행 중에는 대상마다 진행 로그가 나온다.
  - 대상 설명, 준비 단계(`·` 줄: brain 생성, 토큰 발급, 서버 시작, 키 출처), 저장 진행(10개마다 평균 시간)
  - 질문 한 줄씩: `[O]` 맞힘 / `[X]` 틀림, 정답 순위나 대신 나온 사실 id, 첫 검색 시간
  - 마지막 `결과:` 줄에 hit@k · 유의어 · 오탐 · 검색 p50 요약
  - `--brief`: 틀린 질문만 출력 · `--quiet`: 로그 없이 최종 표만 출력
- gbrain 대상은 `gbrain` 실행 파일이 필요하다 (`src/jisu/dashboard/README.md`의 GBrain 설정 참고). 없으면 그 대상만 `skipped`로 남고 나머지는 실행된다.
- 대상마다 임시 폴더에 새 저장소를 만들고, 끝나면 지운다. 사용자의 실제 기억(앱 데이터·`JISU_GBRAIN_HOME`)은 건드리지 않는다.
- gbrain 대상은 호출 하나가 subprocess라서 기본 설정 전체 실행에 약 3분이 걸린다. `--repeat 1`로 줄일 수 있다.
- 종료 코드: 0 = 정상, 2 = 데이터셋·설정 오류이거나 실행된 대상이 없음.

### 측정 대상

| 대상 | 저장 | 검색 |
|---|---|---|
| `json` | upstream `memory.v1.json` | 로컬 키워드 검색 |
| `gbrain` | GBrain (카테고리 = entity + kind) | 앱이 실제로 쓰는 경로: GBrain 순위 → 비면 로컬 키워드 |
| `gbrain-native` | GBrain | GBrain 자체 순위만 (대체 검색 없음) |
| `gbrain-embed` | GBrain + 임베딩 모델 (기본 `google:gemini-embedding-2`, 768차원) | 앱 경로 |
| `gbrain-embed-native` | 위와 같음 | GBrain 자체 순위만 (키워드 + 벡터) |
| `gbrain-http` | GBrain, 상주 HTTP 서버 (`gbrain serve --http`) | 앱 경로 |

`gbrain`과 `gbrain-native`의 차이가 곧 대체 검색이 보탠 몫이다. `gbrain` ↔ `gbrain-embed*`는 임베딩 효과를, `gbrain` ↔ `gbrain-http`는 호출 방식(호출마다 Bun 기동 vs 서버 하나)의 차이를 본다. 새 백엔드는 `subjects.py`에 `Subject`를 구현하고 `SUBJECT_BUILDERS`에 등록하면 된다.

인자 없이 실행하면 기존 3개(`json`, `gbrain`, `gbrain-native`)만 돈다. 아래 대상은 `--subjects`로 직접 골라야 한다.

### 임베딩 비교 (`gbrain-embed*`)

```bash
# 키: GOOGLE_GENERATIVE_AI_API_KEY → GEMINI_API_KEY 순으로 환경변수, 없으면 --env-file(기본 ./.env)에서 이 두 이름만 읽는다

# "임베딩을 켜면 검색이 나아지는가?"를 본다.
# 임베딩 없는 GBrain 2개와 임베딩 켠 GBrain 2개를 나란히 재서, 특히 두 숫자를 비교한다.
#   - paraphrase hit@5: 글자가 안 겹치는 질문("직업" → "백엔드 개발자")을 찾는 비율. 올라가야 좋다.
#   - negative FP: 없는 기억을 물었을 때("고양이 이름") 엉뚱한 답을 내놓는 비율. 내려가야 좋다.
# -native는 대체 키워드 검색 없이 GBrain 순위만 쓴 결과라, 임베딩 자체의 효과를 따로 볼 수 있다.
python -m jisu.bench.memory --subjects gbrain gbrain-native gbrain-embed gbrain-embed-native --detail

# 벡터 길이를 768 대신 1024로 바꿔 같은 시험을 본다. 차원이 정확도·속도에 주는 차이를 확인한다.
python -m jisu.bench.memory --subjects gbrain-embed-native --embedding-dims 1024

# 다른 임베딩 모델로 같은 시험을 본다(모델 이름은 GBrain 형식 provider:model).
python -m jisu.bench.memory --subjects gbrain-embed-native --embedding-model <provider:model>
```

- 키가 없으면 해당 대상만 `skipped`로 남는다.
- 사실 저장·질의마다 임베딩 API를 부른다. 데이터셋 1회 실행(기본 repeat=3)에 약 150~200건, `gemini-embedding-2` 기준 무료 한도 안이다. 생성형 LLM(think·dream)은 부르지 않는다.
- GBrain CLI는 현재 폴더의 `.env`를 읽는다. 그래서 모든 GBrain 호출은 brain 폴더를 작업 폴더로 삼아 실행된다. 저장소 `.env`의 다른 키는 GBrain에 넘어가지 않는다.
- 벡터 칸 길이는 `gbrain init` 때 정해진다. 차원을 바꾸면 새 brain이 필요하다. 벤치는 매번 새 brain을 만드므로 상관없다.

### 호출 방식 비교 (`gbrain-http`)

```bash
# "GBrain을 서버로 띄워 두면 얼마나 빨라지는가?"를 본다.
#   - json:        파일 기반 기준선(가장 빠름)
#   - gbrain:      지금 앱 방식. 호출할 때마다 gbrain 프로세스를 새로 띄운다(Bun 기동 비용 포함)
#   - gbrain-http: gbrain 서버를 한 번 띄워 두고 HTTP로만 호출한다
# 정확도는 gbrain과 gbrain-http가 같아야 하고, 비교할 숫자는 검색·저장 p50(ms)이다.
# 속도 비교가 목적이라 표본을 질문당 5번으로 늘린다.
python -m jisu.bench.memory --subjects json gbrain gbrain-http --repeat 5
```

- 시작할 때 `gbrain auth create`로 토큰을 만들고, 빈 포트에 `gbrain serve --http --surface verbs`를 `127.0.0.1`로 띄운다. 끝나면 서버를 끄고 토큰을 폐기한다. 서버 로그는 brain 폴더의 `serve-http.log`에 남는다.
- 정확도는 `gbrain`과 같아야 한다. 다르면 HTTP 응답 해석(`jisu.dashboard.memory.gbrain_http`)을 의심한다.

### 지표

| 지표 | 정의 (정답 사실 E, 반환 상위 k개 R) |
|---|---|
| hit@1 | R의 첫 결과가 E에 있음 |
| hit@k | R 중 하나라도 E에 있음 |
| recall@k | \|R ∩ E\| / \|E\| |
| MRR | 첫 정답 순위의 역수 평균 (없으면 0) |
| FP | 부정 질의(E가 빈 질의)에 결과가 하나라도 나온 비율 |
| 지연 | 저장 1건 / 검색 1건 / 세션 프롬프트 생성 1회의 p50·p95 (ms), gbrain 초기화 시간 |
| 카테고리 왕복 | 저장한 카테고리가 그대로 읽혀 나온 비율 (카테고리 저장소만) |

정확도는 첫 검색으로 계산한다(결정적). `--repeat`는 지연 표본만 늘린다.

### 데이터셋 작성

`src/jisu/bench/memory/datasets/<이름>.toml` (기본: `ko_desk_v1.toml`, 사실 30개 · 질의 41개)

```toml
schema_version = 1
title = "설명"

[[facts]]
id = "dog_name"
category = "user_fact"   # jisu.persona 카테고리 키
text = "사용자의 강아지 이름은 모찌다"

[[queries]]
id = "dog_multi"
text = "강아지 이름"
expected = ["dog_name"]  # [] = 부정 질의
tags = ["multi_word"]
```

- 태그: `exact`, `short_word`, `multi_word`, `paraphrase`, `english`, `persona`, `promise`, `negative`. 태그는 자유롭게 추가할 수 있고, 태그별 표에 자동으로 나온다.
- 같은 문장(공백·대소문자 무시)의 사실을 두 번 넣을 수 없다. 결과를 텍스트로 사실 id에 되돌려 매칭하기 때문이다.

### 대시보드 연동 지점 (예정)

CLI는 아래 공개 API의 얇은 껍데기다. 대시보드도 이 API만 쓰면 된다.

```python
from jisu.bench.memory import BenchConfig, run_memory_bench, save_report, list_reports, load_report

report = run_memory_bench(BenchConfig(subjects=("json", "gbrain")), on_event=handle)  # 동기, 스레드에서 실행
path = save_report(report)          # 기본 폴더: <앱 데이터>/jisu-bench/memory (JISU_BENCH_RESULTS_DIR로 변경)
summaries = list_reports()          # 최신순: started_at, dataset, k, 대상별 hit@k, skipped
```

- `on_event`는 `BenchEvent(kind, subject, done, total, message)`를 받는다.
  - `kind`: `subject_start` → `subject_ready` | `subject_skipped` → `fact_added`×N → `query_done`×M → `subject_done`
  - 진행 막대는 `done/total`로 그린다.
- 결과 JSON 구조는 `report.py` 모듈 docstring에 있다 (`schema_version` 1). 질의별 상세에는 기대·반환·순위·메모(예: `degraded: no_query_terms`)가 있어서, 실패 질의 화면을 바로 만들 수 있다.
