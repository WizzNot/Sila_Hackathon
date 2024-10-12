from django import forms

class ImageUploadForm(forms.Form):
    images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}), label='Выберите изображения')
