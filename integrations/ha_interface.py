"""Home Assistant Interface - HA OS 音声アシスタント連携

RPi 4B (HA OS) からのHTTP API受付。
常に中隊モード (¥0)。

フロー:
  1. RPi 4Bで音声入力 (Whisper)
  2. HA OS → HTTP POST http://192.168.1.10:8080/api/ask
  3. 本陣で中隊モード起動
  4. 成果物のみ返却 (JSON)
  5. HA OSでTTS (Piper)

使用するエンドポイント:
  POST /api/ask  (main.py の既存エンドポイント)

このファイルは HA OS 側の設定例を提供する。
"""

# Home Assistant configuration.yaml テンプレート
HA_CONFIG_TEMPLATE = """\
# ==============================
# 将軍システム HA OS統合
# ==============================

rest_command:
  ask_company:
    url: "http://192.168.1.10:8080/api/ask"
    method: POST
    headers:
      Content-Type: "application/json"
    payload: >
      {{"question": "{{ question }}"}}
    timeout: 180

  ask_battalion:
    url: "http://192.168.1.10:8080/api/task"
    method: POST
    headers:
      Content-Type: "application/json"
    payload: >
      {{"task": "{{ task }}", "mode": "battalion"}}
    timeout: 300

  shogun_health:
    url: "http://192.168.1.10:8080/api/health"
    method: GET
    timeout: 10

automation:
  # 音声 → 中隊モード
  - id: voice_to_company
    alias: "音声 → 中隊"
    trigger:
      - platform: state
        entity_id: sensor.whisper_transcript
    condition:
      - condition: template
        value_template: "{{ 'クロード' in trigger.to_state.state }}"
    action:
      - service: tts.speak
        data:
          message: "中隊に伝令を送ります"

      - variables:
          question: >
            {{ trigger.to_state.state
               | replace('クロード、', '')
               | replace('クロード', '')
               | trim }}

      - service: rest_command.ask_company
        data:
          question: "{{ question }}"
        response_variable: company_response

      - delay: "00:03:00"

      - service: tts.speak
        data:
          message: "{{ company_response.content.answer }}"

  # ヘルスチェック (毎時)
  - id: hourly_health
    alias: "将軍システムヘルスチェック"
    trigger:
      - platform: time_pattern
        hours: "/1"
    action:
      - service: rest_command.shogun_health
        response_variable: health
      - condition: template
        value_template: "{{ health.content.status != 'ok' }}"
      - service: notify.mobile_app
        data:
          message: "将軍システム異常: {{ health.content.status }}"
"""


def generate_ha_config(output_path: str | None = None) -> str:
    """Generate HA configuration.yaml snippet.

    Args:
        output_path: If provided, write to file.

    Returns:
        Configuration YAML text.
    """
    if output_path:
        from pathlib import Path
        Path(output_path).write_text(HA_CONFIG_TEMPLATE, encoding="utf-8")
    return HA_CONFIG_TEMPLATE


if __name__ == "__main__":
    print(HA_CONFIG_TEMPLATE)
