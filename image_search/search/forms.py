from django import forms

class ImageSearchForm(forms.Form):
    image = forms.ImageField(label="Upload an image")
