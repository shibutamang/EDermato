from django.db import models 

class File(models.Model):
    image = models.ImageField(upload_to = 'images', blank=True, null=True)
    data = models.CharField(max_length=50,null=True)

    def __str__(self):
    	 return self.image