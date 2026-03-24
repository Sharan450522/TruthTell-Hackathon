const axios = require('axios');
const cheerio = require('cheerio');

async function scrapeArticle(url) {
  try {
    const response = await axios.get(url); // Fetch the webpage content
    const html = response.data; // Get the HTML as a string
    const $ = cheerio.load(html); // Load the HTML into Cheerio

    // Extract text within <p> tags
    const paragraphs = [];
    $('p').each((index, element) => {
      paragraphs.push($(element).text());
    });

    return paragraphs.join(' ');
  } catch (error) {
    console.error("Error fetching the article:", error);
    return null;
  }
}

// Example usage
const articleUrl = "https://www.health.harvard.edu/staying-healthy/yoga-benefits-beyond-the-mat";
scrapeArticle(articleUrl).then((articleText) => {
  console.log(articleText);
});
