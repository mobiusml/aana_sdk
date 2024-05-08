from aana.projects.whisper.app import aana_app

asr_deployment = aana_app.get_deployment_app("asr_deployment")
vad_deployment = aana_app.get_deployment_app("vad_deployment")
whisper_app = aana_app.get_main_app()
