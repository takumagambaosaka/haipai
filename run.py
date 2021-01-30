from flask import Flask, render_template, render_template_string, request, redirect
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model


image_model = load_model("image_model.h5", compile=False)
haipai_model = load_model("point_model.h5", compile=False)
point = ["-3901～", "0～-3900", "1～5199", "5200～"]


app = Flask(__name__)

@app.route("/",methods=["GET"])
def index():
 return render_template("index.html")


@app.route("/show_result/", methods=["POST"])
def show_result():
 all_pai = dict.fromkeys(["ドラ数","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37"], 0)

 path = request.form.getlist("pai") #[/img/1.jpeg…]というリスト型のデータ

 haipai = []
 for p in path:
  image = cv2.imread(p)
  image_resize = cv2.resize(image, (75,100))
  new_image = image_resize[np.newaxis]
  pai = np.argmax(image_model.predict(new_image), axis=-1)
  haipai.append(pai) #listの中にndarrayが入っている状態（[[1][2][3]…]）

 if len(haipai) != 14:
  return redirect('/')

 dora = haipai[0]
 haipai = haipai[1:]
 my_hand = []

 #牌の数を数えていく
 #ドラと一致する数を数える
 for hai in haipai:
  if hai == dora:
   all_pai["ドラ数"] +=1
  for h in hai:
   my_hand.append(str(h))
   all_pai[str(h)] += 1
   if str(h) == "10":
    all_pai["ドラ数"] +=1
   elif str(h) == "20":
    all_pai["ドラ数"] +=1
   elif str(h) == "30":
    all_pai["ドラ数"] +=1
   
 list_X = list(all_pai.values())


 dora_number = list_X[0]
 X = np.array(list_X)
 new_X = X[np.newaxis]

#識別したhaipaiから結果を予測
 Y_pred = haipai_model.predict(new_X)
 Y_pred = (Y_pred * 100 + 0.5).astype(int)
 df = pd.DataFrame(data=Y_pred, columns=point)

 return render_template("show_result.html", my_hand=my_hand, dora_number=dora_number, result=df.to_html(index=False))

if __name__ == "__main__":
  app.debug = True
  app.run()