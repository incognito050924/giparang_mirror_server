# giparang_mirror_server

* URLs
  - ../api/analyze : 서버에 분석 요청을 보냄(필수 데이터: Request Body의 image 필드에 사진파일 포함).
  - ../api/result[/YYYY/mm[/DD]] : 월별 데이터 조회(../api/result/YYYY/mm), 일자별 데이터 조회(../api/result/YYYY/mm/DD)
  - ../api/skindata[/id] : 저장된 모든 데이터 조회 혹은 id(int <- auto generated)로 조회

* JSON Data
  - SkinData
  {
    image               : 사진 파일
    measured_at         : 측정 일자(자동 생성)
    is_trained          : 학습 상태(아직 신경 쓸 필요 없음)
    score_pore          : 모공 점수
    score_wrinkle       : 주름 점수
    score_pigmentation  : 홍반 점수
    score_erythema      : 반점 점수
    score_emotion       : 감정 점수
    score_total         : 종합 점수
    comment             : 전문가 코멘트 (아직 필요 없음)
  }
