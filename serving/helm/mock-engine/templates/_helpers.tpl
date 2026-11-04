{{- define "engine.labels" -}}
app.kubernetes.io/name: {{ .Values.serviceName }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: engine
app.kubernetes.io/part-of: tokenforge
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "engine.selector" -}}
app.kubernetes.io/name: {{ .Values.serviceName }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "engine.image" -}}
{{- if .Values.image.digest -}}
{{ .Values.image.repository }}@{{ .Values.image.digest }}
{{- else -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}
{{- end -}}
