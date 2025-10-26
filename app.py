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
import altair as alt
import urllib.parse
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
COOKIE_EXPIRY_MINUTES = 60
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
# ---------- helper fungsi untuk cookies ----------
def save_cookies(cookies, path=COOKIES_FILE):
    data = {
        "cookies": cookies,
        "timestamp": datetime.now()
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_cookies(path=COOKIES_FILE):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    # cek apakah sudah lebih dari 30 menit
    timestamp = data.get("timestamp")
    if timestamp and datetime.now() - timestamp > timedelta(minutes=COOKIE_EXPIRY_MINUTES):
        try:
            os.remove(path)
            print("⚠️ Cookies sudah lebih dari 30 menit — file dihapus otomatis.")
        except Exception as e:
            print(f"⚠️ Gagal hapus cookies: {e}")
        return None

    return data.get("cookies")


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
def get_low_rating_reviews(gmaps_link, max_scrolls=10000):
    options = Options()
    # jangan headless karena beberapa interaksi membutuhkan javascript penuh
    # kamu boleh set headless jika yakin
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # jika ada cookies simpanan, apply dulu
    cookies = load_cookies()
    if cookies:
        try:
            apply_cookies_to_driver(driver, cookies)
            time.sleep(2)
            driver.get("https://www.google.com/maps")
            # cek login
            if not check_logged_in_via_driver(driver, timeout=3):
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
        time.sleep(2)
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
        time.sleep(2)
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

def auto_report_review(row, report_type=None):
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # --- apply cookies ---
    cookies = load_cookies()
    if cookies:
        try:
            apply_cookies_to_driver(driver, cookies)
            time.sleep(2)
            driver.get("https://www.google.com/maps")
            if not check_logged_in_via_driver(driver, timeout=5):
                st.warning("Invalid cookies — login may need to be repeated")
        except Exception as e:
            st.warning(f"Fail apply cookies: {e}")

    try:
        # jika tidak ditentukan manual, ambil hasil semantic prediction
        if not report_type:
            category, _ = classify_report_category(row["Review Text"])
            report_type = category if category in report_categories else report_categories[-1]

        # --- buka tempat di maps ---
        # Prioritas: pakai link yang user input (gmaps_link)
        try:
            if gmaps_link and gmaps_link.strip():
                driver.get(gmaps_link.strip())
            else:
                # fallback ke pencarian manual
                search_url = f"https://www.google.com/maps/search/{row['Place'].replace(' ', '+')}"
                driver.get(search_url)
        except Exception as e:
            st.warning(f"Gagal membuka link Google Maps: {e}")

        time.sleep(5)

        # buka tab review
        try:
            tab = driver.find_element(By.XPATH, "//button[contains(., 'Reviews') or contains(., 'Ulasan')]")
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(3)
        except Exception:
            st.error("tidak bisa buka tab review")
            driver.quit()
            return

        # urutkan peringkat terendah
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

        # scroll biar semua review kebuka
        try:
            scroll_area = driver.find_element(By.XPATH, "//div[contains(@class,'m6QErb') and contains(@class,'DxyBCb')]")
            for _ in range(50):
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_area)
                time.sleep(0.5)
        except Exception:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        # cari user target
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

        # klik titik tiga
        try:
            menu_el = target.find_element(By.XPATH, "./ancestor::div[contains(@class,'jftiEf')]//div[@class='zjA77']")
            driver.execute_script("arguments[0].click();", menu_el)
            time.sleep(2)
        except Exception:
            st.warning("failed to click the three dots")
            driver.quit()
            return

        # klik 'Laporkan ulasan'
        js_click_report = """
        const keywords = ['Report review','Laporkan ulasan','Report','Laporkan'];
        let found = false;
        document.querySelectorAll('*').forEach(el => {
            const txt = (el.innerText || '').trim();
            if (keywords.some(k => txt.includes(k))) {
                try { el.click(); found = true } catch(e) {}
            }
        });
        return found;
        """
        clicked = driver.execute_script(js_click_report)
        if not clicked:
            st.warning("⚠️ Unable to click 'report review'")
            driver.quit()
            return

        st.toast(f"✅ click ‘report review’ to {row['User']}")
        time.sleep(3)

        tabs = driver.window_handles
        if len(tabs) > 1:
            driver.switch_to.window(tabs[-1])
            st.info("🔄 Switch to the report popup tab")
        else:
            st.warning("⚠️ New tab not detected, popup may be in iframe")

        # tunggu popup muncul
        try:
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@role='dialog' or contains(@class,'popup') or contains(@class,'overlay')]")
                )
            )
            st.info("✅ Popup dialog terdeteksi")
        except:
            time.sleep(2)

        # --- klik kategori sesuai struktur baru (aria-label) ---
        
        js_click_category = f"""
        const target = "{report_type}".toLowerCase().trim();

        function sleep(ms) {{
            return new Promise(resolve => setTimeout(resolve, ms));
        }}


        function highlight(el) {{
        el.style.transition = "all 0.3s ease";
        el.style.border = "3px solid red";
        el.style.backgroundColor = "yellow";
        el.scrollIntoView({{behavior:'smooth', block:'center'}});
        }}

        function simulateClick(el) {{
        ['pointerdown','mousedown','mouseup','click'].forEach(evt => {{
            el.dispatchEvent(new MouseEvent(evt, {{ bubbles: true, cancelable: true, view: window }}));
        }});
        }}

        async function runCategoryClick(doc) {{
        const candidates = doc.querySelectorAll('[role="button"], div[role="link"], a, div');

        for (let el of candidates) {{
            let text = (el.innerText || "").toLowerCase().trim();

            // pastikan elemennya hanya mengandung satu kategori, bukan seluruh popup
            if (text.includes(target) && text.length < 60) {{
            highlight(el);
            await sleep(3000); // delay 3 detik
            simulateClick(el);
            return "✅ Clicked category: " + text;
            }}
        }}
        return null;
        }}

        async function start() {{
        let res = await runCategoryClick(document);
        if (res) return res;

        // cek iframe jika ada
        for (let frame of document.querySelectorAll('iframe')) {{
            try {{
            let doc = frame.contentDocument || frame.contentWindow.document;
            res = await runCategoryClick(doc);
            if (res) return res + " (inside iframe)";
            }} catch(e) {{
            continue;
            }}
        }}
        return "⚠️ Category not found: " + target;
        }}

        return await start();
        """




        res_cat = driver.execute_script(js_click_category)
        if res_cat.startswith("✅"):
            st.success(res_cat)
        else:
            st.warning(res_cat)
            with open("last_report_popup_debug.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)

        # --- klik tombol submit ---
        # --- tunggu popup muncul sebelum klik submit ---
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@role='dialog' or contains(@class,'popup') or contains(@class,'overlay')]")
                )
            )
            st.info("✅ Popup dialog terdeteksi, siap klik tombol submit")
        except:
            st.warning("⚠️ Tidak menemukan popup dialog, mencoba lanjut dalam mode halaman penuh...")
            time.sleep(2)

        # --- klik tombol submit / laporkan ---
        js_click_submit = """
        async function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        function highlight(el) {
            el.style.transition = "all 0.3s ease";
            el.style.border = "3px solid red";
            el.style.backgroundColor = "yellow";
            el.scrollIntoView({behavior:'smooth', block:'center'});
        }

        function simulateClick(el) {
            ['pointerdown','mousedown','mouseup','click'].forEach(evt => {
                el.dispatchEvent(new MouseEvent(evt, { bubbles: true, cancelable: true, view: window }));
            });
        }

        async function findAndClickSubmit(root) {
            const keywords = ['submit', 'laporkan', 'send', 'report', 'kirim', 'done', 'selesai'];
            const selectors = [
                'button',
                'div[role="button"]',
                '.VfPpkd-LgbsSe',
                '.VfPpkd-dgl2Hf-ppHlrf-sM5MNb',
                '.VfPpkd-LgbsSe-OWXEXe',
                '.VfPpkd-LgbsSe-OWXEXe-nzrxxc'
            ];

            for (const sel of selectors) {
                const els = root.querySelectorAll(sel);
                for (const el of els) {
                    const txt = (el.innerText || el.ariaLabel || '').toLowerCase().trim();
                    if (keywords.some(k => txt.includes(k))) {
                        highlight(el);
                        await sleep(1000);

                        // --- klik ala user sungguhan ---
                        el.focus();
                        simulateClick(el);

                        // --- coba panggil form handler kalau ada ---
                        const form = el.closest('form');
                        if (form) {
                            try { form.requestSubmit ? form.requestSubmit() : form.submit(); } catch(e) {}
                        }

                        // --- trigger tambahan untuk Google ripple handler ---
                        el.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
                        el.dispatchEvent(new Event('click', { bubbles: true }));
                        
                        await sleep(3500);
                        return "✅ Submit button clicked successfully: " + txt;
                    }
                }
            }

            // recursive shadowRoot
            for (const el of root.querySelectorAll('*')) {
                if (el.shadowRoot) {
                    const res = await findAndClickSubmit(el.shadowRoot);
                    if (res) return res + " (shadowRoot)";
                }
            }

            // cek iframe
            for (const frame of root.querySelectorAll('iframe')) {
                try {
                    const doc = frame.contentDocument || frame.contentWindow.document;
                    const res = await findAndClickSubmit(doc);
                    if (res) return res + " (iframe)";
                } catch(e) {}
            }

            return null;
        }

        async function start() {
            let res = await findAndClickSubmit(document);
            if (res) return res;
            return "⚠️ Tombol submit tidak ditemukan";
        }

        return await start();
        """



        res_submit = driver.execute_script(js_click_submit)
        if res_submit.startswith("✅"):
            st.success(res_submit)
            with open("button.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
        else:
            st.warning(res_submit)
            with open("button.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)


        st.success(f"✅ review {row['User']} successfully reported ({report_type})")

    finally:
        try:
            driver.quit()
        except:
            pass



# ---------- streamlit ui ----------
# st.set_page_config(page_title="Google Maps Review Scraper", layout="wide")
st.title("📍 Google Maps Review Scraper")

if "google_logged" not in st.session_state:
    st.session_state.google_logged = is_cookie_file_present()

st.markdown("## Login Google Account")
if not st.session_state.google_logged:
    st.markdown(
        "Press the button below to open the Chrome browser, then log in to your Google account in the window that appears. After successfully logging in, cookies will be saved."
    )
    if st.button("🔑 Open Browser for login"):
        ok = start_manual_google_login(timeout=300)
        if ok:
            st.success("login successful cookies saved")
            st.session_state.google_logged = True
        else:
            st.error("login failed or timeout, please try again")

else:
    st.success("You are already logged in using stored cookies.n")

st.divider()

# jika belum login tampilkan instruksi dan hentikan
if not st.session_state.google_logged:
    st.info("Please log in before using the scraping feature.")
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

            # Inisialisasi session state untuk pagination
            if "current_page" not in st.session_state:
                st.session_state.current_page = 1

            # Fungsi pindah halaman
            def set_page(p):
                st.session_state.current_page = p

            # Hitung index awal dan akhir
            page = st.session_state.current_page
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_show = df.iloc[start_idx:end_idx]
            st.write(f"Showing {start_idx+1}–{min(end_idx, len(df))} of {len(df)} reviews.")

            # Tampilkan tombol pagination
            st.write("### 📄 Page")
            page_cols = st.columns(min(total_pages, 10))  # Maks 10 tombol per baris

            for i in range(total_pages):
                col = page_cols[i % 10]  # ulang tiap 10 kolom
                with col:
                    page_num = i + 1
                    if st.button(str(page_num), key=f"page_btn_{page_num}"):
                        set_page(page_num)

        else:
            df_show = df
            st.write(f"Showing all {len(df)} reviews.")


        st.markdown("### 💬 Review Table (click 🚨 to mark)")

        if st.button("🚨 REPORT ALL (Auto AI Prediction)", key="report_all"):
            if not df_show.empty:
                reported_count = 0
                for idx, row in df_show.iterrows():
                    category, score = classify_report_category(row["Review Text"])
                    auto_report_review(row, category)  # Langsung report berdasarkan prediksi otomatis
                    reported_count += 1
                st.success(f"✅ Berhasil mereport otomatis {reported_count} review berdasarkan prediksi AI!")
            else:
                st.warning("Tidak ada review untuk direport.")


        df_show = df_show.copy()
        # --- tampilkan tiap review ---
        for idx, row in df_show.iterrows():
            with st.container():
                st.markdown(f"**👤 {row['User']}** — ⭐ {row['Rating']}")
                st.markdown(f"🕒 {row['Date (Parsed)']}  |  {row['Total Reviews']}")
                st.markdown(f"💬 {row['Review Text'] or '_(tidak ada teks)_'}")

                category, score = classify_report_category(row["Review Text"])
                st.markdown(f"**🔖 Prediksi Kategori:** `{category}` ({score}% match)")

                report_choice = st.selectbox(
                    f"📑 Select the type of report for {row['User']}",
                    report_categories,
                    index=report_categories.index(category) if category in report_categories else len(report_categories) - 1,
                    key=f"choice_{idx}"
                )

                # cek apakah review ini sudah pernah direport
                if "reported" not in st.session_state:
                    st.session_state["reported"] = []

                already_reported = any(
                    r["User"] == row["User"] and r["Review Text"] == row["Review Text"]
                    for r in st.session_state["reported"]
                )

                # tombol report otomatis
                if already_reported:
                    st.button("✅ Already Reported", key=f"reported_{idx}", disabled=True)
                else:
                    if st.button("🚨 Automatic Report", key=f"report_{idx}"):
                        try:
                            auto_report_review(row, report_choice)

                            # tambahkan ke daftar reported
                            st.session_state["reported"].append({
                                "User": row["User"],
                                "Review Text": row["Review Text"],
                                "Date": row["Date (Parsed)"],
                                "Kategori Report": report_choice
                            })

                            st.success(f"✅ Review from **{row['User']}** successfully reported and added to the list below!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed Report: {e}")

        if "reported" in st.session_state and st.session_state["reported"]:
            st.divider()
            st.markdown("### 🧾 Reviews that have been submitted")
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
            place_name = st.session_state.place_name or "Lokasi Tidak Diketahui"
            query = urllib.parse.quote_plus(place_name)
            embed_url = f"https://www.google.com/maps?q={query}&output=embed"
            st.markdown(f"📍 **{place_name}**")
            st.components.v1.iframe(embed_url, height=500)

            # --- siapkan selenium ---
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")

            driver = None
            try:
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.get(gmaps_link)
                time.sleep(6)

                # handle redirect (maps.app.goo.gl)
                current_url = driver.current_url
                if "maps.app.goo.gl" in current_url:
                    time.sleep(3)
                    current_url = driver.current_url
                    driver.get(current_url)
                    time.sleep(5)

                # ambil distribusi bintang dari tabel <tr class="BHOKXe">
                rows = driver.find_elements(By.CSS_SELECTOR, "tr.BHOKXe")
                distribusi = {}
                for r in rows:
                    label = r.get_attribute("aria-label")  # contoh: "5 stars, 182 reviews"
                    if label:
                        try:
                            bintang = int(label.split()[0])
                            jumlah = int(label.split(",")[1].split()[0])
                            distribusi[bintang] = jumlah
                        except Exception:
                            continue

            finally:
                # pastikan driver ditutup jika dibuat
                if driver is not None:
                    try:
                        driver.quit()
                    except Exception:
                        pass

            # --- tampilkan distribusi ---
            if distribusi:
                st.markdown(f"### 📊 **Rating Distribution {place_name}**")
                df_dist = (
                    pd.Series(distribusi)
                    .reindex([5, 4, 3, 2, 1], fill_value=0)
                    .rename_axis("Rating")
                    .reset_index(name="Jumlah Review")
                )
                st.dataframe(df_dist, use_container_width=True, hide_index=True)

                # --- definisi warna berdasarkan rating ---
                warna = {
                    5: "#4CAF50",  # hijau
                    4: "#8BC34A",  # hijau muda
                    3: "#FFC107",  # kuning
                    2: "#FF9800",  # oranye
                    1: "#F44336",  # merah
                }
                df_dist["Warna"] = df_dist["Rating"].map(warna)

                # --- visualisasi pakai altair ---
                chart = (
                    alt.Chart(df_dist)
                    .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
                    .encode(
                        x=alt.X("Rating:O", sort="descending", axis=alt.Axis(title="Bintang")),
                        y=alt.Y("Jumlah Review:Q", axis=alt.Axis(title="Jumlah")),
                        color=alt.Color("Warna:N", scale=None, legend=None),
                        tooltip=["Rating", "Jumlah Review"]
                    )
                    .properties(height=300)
                )

                st.altair_chart(chart, use_container_width=True)

                # --- hitung rata-rata rating ---
                total_review = df_dist["Jumlah Review"].sum()
                if total_review > 0:
                    avg_rating = (df_dist["Rating"] * df_dist["Jumlah Review"]).sum() / total_review
                    st.markdown(
                        f"<h4 style='color:#FFD700;'>⭐ Rata-rata Rating: {avg_rating:.2f}</h4>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("**⭐ Average Rating: {place_name} **")
            else:
                st.info("Tidak ada data penyebaran rating yang ditemukan.")

        except Exception as e:
            st.warning(f"Gagal memuat peta atau rating: {e}")

    # --- bagian review tetap ---
    if "df_reviews" in st.session_state and not st.session_state.df_reviews.empty:
        import altair as alt
        import pandas as pd

        st.markdown("### 💢 Negative Review Distribution (1–2 Stars) - {place_name}")
        df = st.session_state.df_reviews

        # --- ambil hanya rating 1 dan 2 ---
        rating_counts = (
            df["Rating"].value_counts()
            .reindex([2, 1], fill_value=0)
        )

        summary_df = rating_counts.rename_axis("Rating").reset_index(name="Jumlah Review")

        # --- warna khusus ---
        warna = {
            2: "#FFC107",  # kuning
            1: "#F44336",  # merah
        }

        summary_df["Warna"] = summary_df["Rating"].map(warna)
        # --- PIE CHART ---
        pie_chart = (
            alt.Chart(summary_df)
            .mark_arc(outerRadius=120, innerRadius=50)
            .encode(
                theta="Jumlah Review:Q",
                color=alt.Color("Warna:N", scale=None, legend=None),
                tooltip=["Rating", "Jumlah Review"]
            )
            .properties(height=350)
        )

        st.dataframe(summary_df.drop(columns=["Warna"]), use_container_width=True, hide_index=True)
        st.altair_chart(pie_chart, use_container_width=True)

        total_reviews = rating_counts.sum()
        st.markdown(f"**Overall Rating (1–2 stars): {place_name}** {total_reviews}")

    else:
        st.info("Belum ada data review untuk diringkas.")
