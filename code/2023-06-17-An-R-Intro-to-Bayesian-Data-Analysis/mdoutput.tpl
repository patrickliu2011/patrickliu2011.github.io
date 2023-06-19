{% extends 'markdown/index.md.j2' %}

{% block input %}
{{ super() }}
{% endblock input %}

{% block in_prompt %}
In[{{cell.execution_count if cell.execution_count else ' '}}]:
{% endblock in_prompt %}

{% block output_prompt %}
Out[{{cell.execution_count}}]:
{%- endblock output_prompt %}

{% block stream %}
```
{{ output.text }}
```
{% endblock stream %}

{% block data_text %}
```
{{ output.data['text/plain'] }}
```
{% endblock data_text %}

{% block traceback_line %}
{{ line | strip_ansi }}
{% endblock traceback_line %}

{% block data_html %}
{{ output.data['text/markdown'] }}
{% endblock data_html %}