from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from melo.api import TTS
import torch
import uvicorn
import re
import json

app = FastAPI()

class SynthesizeRequest(BaseModel):
    text: str
    speaker: str = "KR"
    speed: float = 1.0
    output_path: str = "output.wav"

def preprocess_text(text):
    # 특수 문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 양쪽 공백 제거
    text = text.strip()
    
    return text

def preprocess_json_data(json_data):
    # JSON 문자열로 변환
    json_string = json.dumps(json_data)
    
    # 제어 문자 제거
    json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string)
    
    # JSON 객체로 다시 파싱
    preprocessed_data = json.loads(json_string)
    
    return preprocessed_data

@app.post("/tts")
def convert_text_to_speech(request: SynthesizeRequest):
    try:
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        model = TTS(language=request.speaker, device=device)
        speaker_ids = model.hps.data.spk2id
        
        # 입력된 JSON 데이터 전처리
        preprocessed_data = preprocess_json_data(request.dict())
        
        # 전처리된 데이터에서 필요한 값 추출
        preprocessed_text = preprocess_text(preprocessed_data['text'])
        speaker = preprocessed_data['speaker']
        speed = preprocessed_data['speed']
        output_path = preprocessed_data['output_path']

        model.tts_to_file(preprocessed_text, speaker_ids[speaker], output_path, speed=speed)

        return {"message": "Text converted to speech", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7778)