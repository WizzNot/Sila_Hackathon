from django.shortcuts import render


def homepage(request):
    template = "homepage/home.html"
    return render(request, template_name=template)


def upload(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        images = request.FILES.getlist('images')
        for image in images:
            print(image)
        return render(request, 'homepage/upload_success.html')
    return render(request, 'homepage/upload_fail.html')