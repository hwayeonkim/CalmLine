# 변환 음성의 짧은 뒷문장 누락 수정 (2026-10-17)

## 재현과 원인

배포 v20의 난방 샘플을 sample cache OFF로 실행했다. 화면에 `This is the fourth time. I've called.`가 있었지만,
`/api/voice/<session>/1`의 서버 원본 WAV(약1.43초)에는 `This is the fourth time.`만 있었다. 재실행에서도 동일했다.
이번 현상은 STT 누락이나 영상 편집 문제가 아니라 native-audio 낭독에서 뒤 문장이 생략된 경우다.
원음·변환음이 섞인 전체 녹음에 대한 자동 전사는 누락 판별에 적합하지 않았다.

## 변경

- Vertex native-audio에 여러 문장을 한 turn으로 보내는 대신, 문장 단위로 순서대로 읽힌다.
- 같은 reader의 다른 요청이 중간에 끼어들지 않도록 전체 스크립트에도 lock을 둔다.
- 주소의 `21st. Street`, `Apt.` 같은 표기는 분할에서 보호한다.
- 출력 전사에서 빠진 구절이 의심되면 audit에 기록하고 해당 음성을 샘플 캐시에 저장하지 않는다.
- 기존에 잘린 PCM을 다시 쓰지 않도록 voice cache에 `sentence-turns-v2` revision을 추가했다.
  기존 STT와 card 캐시는 유지하며, 예전 음성 데이터는 삭제하지 않는다.

Live가 반환하는 출력 전사만으로 실제 음성 누락을 확정하지 않는다. 실제 질문 PCM은 완전했지만
출력 전사가 `Could you`에서 끝난 실행도 관측했다. 따라서 이 신호로 이미 재생된 음성을 취소하거나
대체 음성을 중복 재생하지 않는다. 이 검사는 모든 단어·숫자의 음성 충실도를 보증하지 않는다.

## 검증

`python -m unittest discover -s tests -p 'test_*voice*.py' -v` (12개)

동일 프로젝트의 Vertex native-audio에서 실제 사용 pace인 fast로 문제 문장, 난방/주소 문장, 질문 문장을
실행했다. 별도 로컬 Whisper 전사로 생성 PCM을 확인했고 문제 문장에서 `I've called`가 남았다.
일반 pace도 같은3개 문장을 확인했다. 소규모 회귀 확인이며 정량 성능 벤치마크는 아니다.
여러 문장을 나눠 생성하므로 문장 사이 접속·생성 대기 시간이 추가될 수 있다.

배포 후에는 공개 `/demo`에서 캐시 OFF로 재실행한 문장별 WAV와 영상 오디오를 대조한다.
