import streamlit as st
st.set_page_config(page_title="Google Maps Review Scraper", layout="wide")

from sentence_transformers import SentenceTransformer, util
import time
import os
import pickle
import io
import re
import emoji
import pandas as pd
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz, process

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
from datetime import datetime, timedelta

# ---------- konfigurasi ----------
COOKIES_FILE = "gmaps_cookies.pkl"
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

report_categories = [
    "Off topic",
    "Spam",
    "Conflict of interest",
    "Profanity",
    "Bullying or harassment",
    "Discrimination or hate speech",
    "Personal information",
    "Not helpful"
]


# ---------- helper fungsi untuk cookies ----------
def save_cookies(cookies, path=COOKIES_FILE):
    with open(path, "wb") as f:
        pickle.dump(cookies, f)

def load_cookies(path=COOKIES_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def is_cookie_file_present():
    return os.path.exists(COOKIES_FILE)

# ---------- fungsi untuk memulai browser agar user login manual ----------
def start_manual_google_login(timeout=300):
    """
    buka browser chrome non headless ke halaman login google
    user harus menyelesaikan login manual termasuk 2fa atau captcha
    ketika url berubah keluar dari accounts.google.com atau jika avatar muncul
    maka cookies disimpan
    """
    options = Options()
    # jangan headless karena user harus berinteraksi
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get("https://accounts.google.com/signin/v2/identifier")
        st.info("browser terbuka silakan login di jendela yang muncul selesaikan semua 2fa atau captcha jika muncul")
        wait = WebDriverWait(driver, 5)
        start = time.time()
        while True:
            current_url = driver.current_url
            # jika url tidak lagi berada di accounts.google.com besar kemungkinan sudah login
            if "accounts.google.com" not in current_url:
                # simpan cookies
                cookies = driver.get_cookies()
                save_cookies(cookies)
                driver.quit()
                return True
            # cek juga indikator avatar presence
            try:
                avatar = driver.find_elements(By.XPATH, "//img[contains(@alt,'Google Account') or contains(@alt,'Foto profil')]")
                if avatar:
                    cookies = driver.get_cookies()
                    save_cookies(cookies)
                    driver.quit()
                    return True
            except Exception:
                pass

            if time.time() - start > timeout:
                # timeout menunggu login manual
                driver.quit()
                return False
            time.sleep(1)
    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        st.error(f"gagal membuka browser untuk login {e}")
        return False


# ---------- semantic model setup ----------
@st.cache_resource
def load_semantic_model():
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    category_embeddings = model.encode(report_categories, convert_to_tensor=True)
    return model, category_embeddings

model, category_embeddings = load_semantic_model()

# ---------- helper untuk memuat cookies ke driver baru ----------
def apply_cookies_to_driver(driver, cookies):
    """
    driver harus sudah mengunjungi domain utama google dulu
    kemudian kita tambahkan cookies satu per satu
    """
    driver.get("https://www.google.com")
    # hapus cookies default agar terpakai cookies kita
    driver.delete_all_cookies()
    for c in cookies:
        # selenium add_cookie mengharapkan dict dengan nama domain path value
        # some keys like sameSite may cause issues jadi kita filter
        cookie = {}
        for k in ("name", "value", "path", "domain", "secure", "httpOnly", "expiry"):
            if k in c:
                cookie[k] = c[k]
        try:
            driver.add_cookie(cookie)
        except Exception:
            # jika gagal tambahkan expiry atau properti lain mungkin problem
            try:
                # coba tanpa expiry
                cookie2 = {k: cookie[k] for k in cookie if k != "expiry"}
                driver.add_cookie(cookie2)
            except Exception:
                pass
    driver.refresh()
    time.sleep(2)

def check_logged_in_via_driver(driver, timeout=10):
    """
    coba deteksi apakah sudah login dengan melihat avatar atau tombol sign out
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            # avatar indicator on google main
            avatars = driver.find_elements(By.XPATH, "//img[contains(@alt,'Google Account') or contains(@aria-label,'Profile') or contains(@alt,'Foto profil')]")
            if avatars:
                return True
            # atau tombol sign out di accounts
            signout = driver.find_elements(By.XPATH, "//*[contains(text(),'Sign out') or contains(text(),'Keluar')]")
            if signout:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def classify_report_category(review_text):
    if not review_text or len(review_text.strip()) < 3:
        return "Other", 0.0

    text_embedding = model.encode(review_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(text_embedding, category_embeddings)
    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[0][best_idx].item()

    return report_categories[best_idx], round(best_score * 100, 2)


def clean_review_text_en(text):
    if not text:
        return ""
    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words).strip()

# ---------- helper parse tanggal relatif ----------
def parse_relative_date(text):
    text = (text or "").lower().strip()
    now = datetime.now()
    patterns = [
        (r"(\d+)\s+day", "days"),
        (r"(\d+)\s+week", "weeks"),
        (r"(\d+)\s+month", "months"),
        (r"(\d+)\s+year", "years"),
    ]
    for pattern, unit in patterns:
        match = re.search(pattern, text)
        if match:
            num = int(match.group(1))
            if unit == "days":
                return (now - timedelta(days=num)).strftime("%Y-%m-%d")
            elif unit == "weeks":
                return (now - timedelta(weeks=num)).strftime("%Y-%m-%d")
            elif unit == "months":
                return (now - timedelta(days=30 * num)).strftime("%Y-%m-%d")
            elif unit == "years":
                return (now - timedelta(days=365 * num)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(text, "%B %Y").strftime("%Y-%m-%d")
    except Exception:
        return text

# ---------- fungsi scraping yang memanfaatkan cookies ----------
def get_low_rating_reviews(gmaps_link, max_scrolls=100):
    options = Options()
    # jangan headless karena beberapa interaksi membutuhkan javascript penuh
    # kamu boleh set headless jika yakin
    # options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # jika ada cookies simpanan, apply dulu
    cookies = load_cookies()
    if cookies:
        try:
            apply_cookies_to_driver(driver, cookies)
            # cek login
            if not check_logged_in_via_driver(driver, timeout=5):
                st.warning("cookies ditemukan tapi sepertinya tidak valid atau sudah kadaluarsa silakan login ulang")
        except Exception as e:
            st.warning(f"gagal apply cookies {e}")

    # lalu buka maps
    driver.get(gmaps_link)
    time.sleep(5)

    # --- Auto-detect place name ---
    try:
        place_name = driver.find_element(By.XPATH, "//h1[contains(@class, 'DUwDvf')]").text.strip()
    except Exception:
        place_name = "Unknown_Place"

    # --- Click Reviews tab ---
    try:
        review_tab = driver.find_element(By.XPATH, "//button[contains(., 'Reviews') or contains(., 'Ulasan')]")
        driver.execute_script("arguments[0].click();", review_tab)
        time.sleep(4)
    except Exception:
        pass

    # --- Sort by lowest rating ---
    try:
        sort_button = driver.find_element(By.XPATH, "//button[contains(., 'Sort') or contains(., 'Urutkan')]")
        driver.execute_script("arguments[0].click();", sort_button)
        time.sleep(1)
        lowest = driver.find_elements(By.XPATH, "//*[contains(text(), 'Lowest rating') or contains(text(), 'Peringkat terendah')]")
        for opt in lowest:
            try:
                driver.execute_script("arguments[0].click();", opt)
                break
            except Exception:
                continue
        time.sleep(3)
    except Exception:
        pass

    # --- Scroll efficiently ---
    try:
        scrollable_div = driver.find_element(By.XPATH, "//div[contains(@class,'m6QErb') and contains(@class,'DxyBCb')]")
    except Exception:
        scrollable_div = None

    if scrollable_div:
        last_height = 0
        same_count = 0
        for i in range(max_scrolls):
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(0.5)
            new_height = driver.execute_script("return arguments[0].scrollTop", scrollable_div)
            if new_height == last_height:
                same_count += 1
                if same_count >= 2:
                    break
            else:
                same_count = 0
            last_height = new_height
    else:
        # fallback scroll page
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

    # --- Extract all reviews ---
    blocks = driver.find_elements(By.CLASS_NAME, "jftiEf")
    data = []

    for rb in blocks:
        try:
            rating_text = rb.find_element(By.CLASS_NAME, "kvMYJc").get_attribute("aria-label")
            rating = rating_text.split()[0] if rating_text else ""
        except Exception:
            rating = ""
        try:
            more_button = rb.find_element(By.CLASS_NAME, "w8nwRe")
            driver.execute_script("arguments[0].click();", more_button)
            time.sleep(0.2)
        except Exception:
            pass
        try:
            text = rb.find_element(By.CLASS_NAME, "wiI7pd").text.strip()
        except Exception:
            text = ""
        clean_text = clean_review_text_en(text)
        try:
            user = rb.find_element(By.CLASS_NAME, "d4r55").text
        except Exception:
            user = ""
        try:
            date_txt = rb.find_element(By.CLASS_NAME, "rsqaWe").text
            date_parsed = parse_relative_date(date_txt)
        except Exception:
            date_txt = ""
            date_parsed = ""
        try:
            total_reviews = rb.find_element(By.CLASS_NAME, "RfnDt").text
        except Exception:
            total_reviews = ""
        try:
            rating_value = float(rating)
        except Exception:
            rating_value = 0
        if rating_value in [1.0, 2.0]:
            data.append({
                "Place": place_name,
                "User": user,
                "Total Reviews": total_reviews,
                "Rating": rating_value,
                "Date (Raw)": date_txt,
                "Date (Parsed)": date_parsed,
                "Review Text": clean_text
            })

    driver.quit()
    df = pd.DataFrame(data)
    df["Place"] = place_name
    return df, place_name

def auto_report_review(row, report_type="Spam"):
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # --- apply cookies ---
    cookies = load_cookies()
    if cookies:
        try:
            apply_cookies_to_driver(driver, cookies)
            if not check_logged_in_via_driver(driver, timeout=5):
                st.warning("cookies tidak valid — login mungkin perlu diulang")
        except Exception as e:
            st.warning(f"gagal apply cookies: {e}")

    try:
        # --- buka tempat ---
        search_url = f"https://www.google.com/maps/search/{row['Place'].replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(5)

        # --- buka tab review ---
        try:
            tab = driver.find_element(By.XPATH, "//button[contains(., 'Reviews') or contains(., 'Ulasan')]")
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(3)
        except Exception:
            st.error("tidak bisa buka tab review")
            driver.quit()
            return

        # --- urutkan peringkat terendah ---
        try:
            sort_button = driver.find_element(By.XPATH, "//button[contains(., 'Sort') or contains(., 'Urutkan')]")
            driver.execute_script("arguments[0].click();", sort_button)
            time.sleep(1)
            lowest = driver.find_elements(By.XPATH, "//*[contains(text(), 'Lowest rating') or contains(text(), 'Peringkat terendah')]")
            for opt in lowest:
                try:
                    driver.execute_script("arguments[0].click();", opt)
                    break
                except:
                    continue
            time.sleep(3)
        except Exception:
            pass

        # --- scroll ulasan ---
        try:
            scroll_area = driver.find_element(By.XPATH, "//div[contains(@class,'m6QErb') and contains(@class,'DxyBCb')]")
        except Exception:
            scroll_area = None

        if scroll_area:
            for _ in range(5):
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_area)
                time.sleep(1)
        else:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        # --- cari user ---
        users = driver.find_elements(By.CSS_SELECTOR, ".d4r55")
        target = None
        for u in users:
            if row["User"].lower() in u.text.lower():
                target = u
                break
        if not target:
            st.warning(f"user {row['User']} tidak ditemukan")
            driver.quit()
            return

        driver.execute_script("arguments[0].scrollIntoView({behavior:'smooth',block:'center'});", target)
        time.sleep(1)

        # --- klik titik tiga ---
        try:
            menu_el = target.find_element(By.XPATH, "./ancestor::div[contains(@class,'jftiEf')]//div[@class='zjA77']")
            driver.execute_script("arguments[0].click();", menu_el)
            time.sleep(2)
        except Exception:
            st.warning("gagal klik titik tiga")
            driver.quit()
            return

        # --- klik laporkan ulasan ---
        js_click_report = """
        const keywords = ['Report review','Laporkan ulasan','Report','Laporkan'];
        let clicked = false;
        document.querySelectorAll('*').forEach(el => {
            const txt = el.innerText ? el.innerText.trim() : '';
            if (keywords.some(k => txt.includes(k))) {
                try { el.click(); clicked = true } catch(e) {}
            }
        });
        return clicked;
        """
        clicked = driver.execute_script(js_click_report)

        if not clicked:
            st.warning("⚠️ tidak bisa klik 'laporkan ulasan'")
            driver.quit()
            return

        st.toast(f"✅ klik 'laporkan ulasan' untuk {row['User']}")
        time.sleep(4)

      # === klik kategori report (versi sesuai struktur Coba dulu.html / Google Maps modern) ===
  # === klik kategori report (DOM modern, lebih tangguh & adaptif) ===
        try:
            # Tunggu popup muncul
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@role='dialog' or contains(@class,'popup') or contains(@class,'overlay')]")
                )
            )
            time.sleep(1.5)

            target_text = report_type.lower().strip()
            clicked_flag = False

            js_code = f"""
            const target = "{target_text}";
            const selectors = ['a', 'div', 'span', 'button', '[role="option"]', '[role="radio"]', 'label'];
            for (const sel of selectors) {{
                const elements = document.querySelectorAll(sel);
                for (const el of elements) {{
                    const txt = (el.innerText || '').toLowerCase().trim();
                    if (txt.includes(target)) {{
                        el.scrollIntoView({{behavior:'smooth', block:'center'}});
                        el.click();
                        return "✅ Berhasil klik kategori: " + txt;
                    }}
                }}
            }}
            // coba klik radio input kalau ada
            const radios = document.querySelectorAll('input[type="radio"]');
            for (const r of radios) {{
                const lbl = r.closest('label');
                const txt = lbl ? (lbl.innerText || '').toLowerCase().trim() : '';
                if (txt.includes(target)) {{
                    lbl.scrollIntoView({{behavior:'smooth', block:'center'}});
                    lbl.click();
                    return "✅ Berhasil klik radio kategori: " + txt;
                }}
            }}
            return "⚠️ Tidak menemukan kategori yang cocok";
            """

            result = driver.execute_script(js_code)

            if result.startswith("✅"):
                st.success(result)
                clicked_flag = True
            else:
                st.warning(result)

            if not clicked_flag:
                popup = driver.find_element(By.XPATH, "//div[@role='dialog' or contains(@class,'popup') or contains(@class,'overlay')]")
                html = popup.get_attribute("outerHTML")
                with open("last_report_popup.html", "w", encoding="utf-8") as f:
                    f.write(html)
                st.warning("⚠️ kategori tidak ditemukan — popup disimpan ke last_report_popup.html")

        except Exception as e:
            st.error(f"❌ gagal klik kategori '{report_type}': {e}")

        # === klik tombol submit (lebih fleksibel) ===
        try:
            time.sleep(1)
            js_submit = """
            const possible = ['submit','send','kirim','laporkan','report'];
            const buttons = document.querySelectorAll('button, div[role="button"], span');
            for (const btn of buttons) {
                const txt = (btn.innerText || '').toLowerCase();
                if (possible.some(k => txt.includes(k))) {
                    btn.scrollIntoView({behavior:'smooth', block:'center'});
                    btn.click();
                    return "✅ Tombol submit diklik: " + txt;
                }
            }
            return "⚠️ Tidak menemukan tombol submit di DOM";
            """
            res = driver.execute_script(js_submit)

            if res.startswith("✅"):
                st.success(res)
            else:
                st.warning(res)
        except Exception as e:
            st.warning(f"⚠️ gagal klik submit: {e}")




        # --- klik submit ---
        try:
            submit_btn = driver.find_element(By.CSS_SELECTOR, "div.VfPpkd-RLmnJb")
            driver.execute_script("arguments[0].scrollIntoView({behavior:'smooth',block:'center'});", submit_btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", submit_btn)
            st.success("✅ tombol submit berhasil diklik")
        except Exception as e:
            st.warning(f"⚠️ gagal klik submit: {e}")

        time.sleep(2)
        st.success(f"✅ review {row['User']} sudah dilaporkan")

    finally:
        try:
            driver.quit()
        except:
            pass



# ---------- streamlit ui ----------
# st.set_page_config(page_title="Google Maps Review Scraper", layout="wide")
st.title("📍 Google Maps Review Scraper (login manual sekali)")

if "google_logged" not in st.session_state:
    st.session_state.google_logged = is_cookie_file_present()

st.markdown("## langkah login")
if not st.session_state.google_logged:
    st.markdown(
        "tekan tombol di bawah untuk membuka browser chrome lalu login ke akun google kamu di jendela yang muncul setelah berhasil login cookies akan disimpan"
    )
    if st.button("🔑 buka browser untuk login google"):
        ok = start_manual_google_login(timeout=300)
        if ok:
            st.success("login berhasil cookies tersimpan kamu bisa lanjut ke scraping")
            st.session_state.google_logged = True
        else:
            st.error("login gagal atau timeout silakan coba lagi")

else:
    st.success("kamu sudah login menggunakan cookies tersimpan")

st.divider()

# jika belum login tampilkan instruksi dan hentikan
if not st.session_state.google_logged:
    st.info("silakan login dulu sebelum menggunakan fitur scraping")
    st.stop()

# setelah login tampilkan fitur scraping yang sama dengan skripmu
gmaps_link = st.text_input("🔗 Google Maps Link:")

col1, col2 = st.columns([2, 1])

with col1:
    if "df_reviews" not in st.session_state:
        st.session_state.df_reviews = pd.DataFrame()
        st.session_state.place_name = ""

    if st.button("🚀 Start Scraping"):
        if gmaps_link:
            with st.spinner("Fetching low-rating reviews... please wait a few minutes."):
                try:
                    df, place_name = get_low_rating_reviews(gmaps_link)
                except Exception as e:
                    st.error(f"gagal scraping {e}")
                    df = pd.DataFrame()
                    place_name = ""
            if not df.empty:
                st.session_state.df_reviews = df
                st.session_state.place_name = place_name
                st.success(f"✅ Collected {len(df)} low-rating reviews from **{place_name}**")
            else:
                st.warning("No 1★ or 2★ reviews found.")
        else:
            st.error("Please input a valid Google Maps link.")

    df = st.session_state.df_reviews

    if not df.empty:
        st.divider()
        st.subheader(f"📊 Low-Rating Reviews from: {st.session_state.place_name}")

        per_page = st.selectbox("Show reviews per page:", [10, 25, 100, "All"])

        if per_page != "All":
            per_page = int(per_page)
            total_pages = (len(df) - 1) // per_page + 1
            page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_show = df.iloc[start_idx:end_idx]
            st.write(f"Showing {start_idx+1}–{min(end_idx, len(df))} of {len(df)} reviews.")
        else:
            df_show = df
            st.write(f"Showing all {len(df)} reviews.")

        st.markdown("### 💬 Review Table (click 🚨 to mark)")

        df_show = df_show.copy()
        for idx, row in df_show.iterrows():
            with st.container():
                st.markdown(f"**👤 {row['User']}** — ⭐ {row['Rating']}")
                st.markdown(f"🕒 {row['Date (Parsed)']}  |  {row['Total Reviews']}")
                st.markdown(f"💬 {row['Review Text'] or '_(tidak ada teks)_'}")
                category, score = classify_report_category(row["Review Text"])
                st.markdown(f"**🔖 Prediksi Kategori:** `{category}` ({score}% match)")
                report_choice = st.selectbox(
                    f"📑 Pilih jenis report untuk {row['User']}",
                    report_categories,
                    index=report_categories.index(category) if category in report_categories else len(report_categories) - 1,
                    key=f"choice_{idx}"
                )
                if st.button(f"🚨 Report Otomatis", key=f"report_{idx}"):
                    auto_report_review(row, report_choice)

        if "reported" in st.session_state and st.session_state["reported"]:
            st.divider()
            st.markdown("### 🧾 Review yang Ditandai untuk Dilaporkan")
            st.dataframe(pd.DataFrame(st.session_state["reported"]), use_container_width=True, hide_index=True)

        place_filename = st.session_state.place_name.replace(" ", "_").replace("/", "_")
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button(
            "💾 Download Excel File",
            buffer,
            file_name=f"low_rating_reviews_{place_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with col2:
    if gmaps_link:
        st.markdown("### 🗺️ Google Maps View")
        try:
            import urllib.parse
            place_name = st.session_state.place_name or "Lokasi Tidak Diketahui"
            query = urllib.parse.quote_plus(place_name)
            embed_url = f"https://www.google.com/maps?q={query}&output=embed"
            st.markdown(f"📍 **{place_name}**")
            st.components.v1.iframe(embed_url, height=500)
        except Exception as e:
            st.warning(f"Gagal memuat peta: {e}")

    if "df_reviews" in st.session_state and not st.session_state.df_reviews.empty:
        st.markdown("### ⭐ Ringkasan Rating")
        df = st.session_state.df_reviews
        rating_counts = df["Rating"].value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)
        total_reviews = rating_counts.sum()
        avg_rating = (rating_counts.index.to_numpy(dtype=float) * rating_counts.values).sum() / total_reviews if total_reviews > 0 else 0
        summary_df = rating_counts.rename_axis("Rating").reset_index(name="Jumlah Review")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.bar_chart(rating_counts.sort_index(ascending=True))
        st.markdown(f"**Total Review:** {total_reviews}  \n**Rata-rata Rating:** ⭐ {avg_rating:.2f}")
    else:
        st.info("Belum ada data review untuk diringkas.")
